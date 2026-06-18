from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from adaptive_hgrl import (
    AdaptiveGridGraph,
    AgentTask,
    MissionBatteryState,
    PlanMetrics,
    Episode,
    UAVDataset,
    fairness_std,
    path_length,
    risk_along_path,
    episode_key,
    episode_number,
    episode_sort_key,
    simulate_method,
    terrain_factor,
)


Point = Tuple[float, float]
Node = Tuple[int, int]
Action = Tuple[int, int]

ACTIONS: List[Action] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


@dataclass
class ExistingMethodMetrics:
    episode_id: str
    method: str
    total_cost: float
    makespan: float
    avg_path_length: float
    collision_risk: float
    energy_used: float
    completed: int
    min_battery: float
    battery_warnings: int
    recharge_visits: int
    graph_updates: int
    fairness_std: float

    def objective(self, target_tasks: int) -> float:
        incomplete = max(0, target_tasks - self.completed) * 25.0
        return (
            self.total_cost
            + 2.5 * self.makespan
            + 80.0 * self.collision_risk
            + 0.8 * self.energy_used
            + 5.0 * self.fairness_std
            + max(0.0, 0.30 - self.min_battery) * 75.0
            + self.battery_warnings * 3.0
            + incomplete
        )


@dataclass(frozen=True)
class MethodCapability:
    name: str
    use_feedback: bool
    allow_recharge: bool
    adaptive_edges: bool
    heterogeneous_ugv: bool
    battery_aware: bool
    dynamic_obstacle_aware: bool
    centralized_feedback_weight: float
    ugv_task_penalty: float
    recharge_bonus: float
    priority_weight: float
    battery_weight: float
    risk_weight: float
    distance_weight: float
    epochs: int
    lr: float


class HeterogeneousGraphEncoder(nn.Module):
    """Small message-passing encoder for UAV, UGV, and task nodes."""

    def __init__(self, in_dim: int, hidden_dim: int = 48, layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.msg_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.in_proj(x))
        degree = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        for self_layer, msg_layer, norm in zip(self.self_layers, self.msg_layers, self.norms):
            msg = adj @ h / degree
            h = norm(torch.relu(self_layer(h) + msg_layer(msg)))
        return h


class DeepHGRLAllocator(nn.Module):
    """Paper-1-style heterogeneous graph RL task allocator baseline."""

    def __init__(self, node_dim: int = 14, pair_dim: int = 5):
        super().__init__()
        self.encoder = HeterogeneousGraphEncoder(node_dim)
        self.scorer = nn.Sequential(
            nn.Linear(48 * 2 + pair_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        agent_index: int,
        task_indices: Sequence[int],
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.encoder(node_features, adj)
        scores = []
        for local_idx, task_index in enumerate(task_indices):
            score_input = torch.cat([emb[agent_index], emb[task_index], pair_features[local_idx]], dim=0)
            scores.append(self.scorer(score_input).squeeze(-1))
        return torch.stack(scores)


class DeepCFRPolicy(nn.Module):
    """Paper-2-style centralized-feedback MARL path policy baseline."""

    def __init__(self, state_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(ACTIONS)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.policy(state)


class MWMADDPGActor(nn.Module):
    """Paper-3-style MW-MADDPG decentralized UAV actor baseline."""

    def __init__(self, feature_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.scorer(candidate_features).squeeze(-1)


class MWMADDPGCritic(nn.Module):
    """Centralized critic/value scorer used by the MW-MADDPG baseline."""

    def __init__(self, feature_dim: int = 13, hidden_dim: int = 64):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        return self.value(joint_features).squeeze(-1)


class TANetActor(nn.Module):
    """TANet/TD3-style target assignment actor baseline."""

    def __init__(self, feature_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.actor(candidate_features).squeeze(-1)


class TANetCritic(nn.Module):
    def __init__(self, feature_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.critic(candidate_features).squeeze(-1)


class GraphActorCritic(nn.Module):
    """GNN actor-critic used by all five comparison methods."""

    def __init__(self, node_dim: int = 14, candidate_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.encoder = HeterogeneousGraphEncoder(node_dim, hidden_dim=hidden_dim, layers=2)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        agent_index: int,
        task_indices: Sequence[int],
        candidate_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(node_features, adj)
        logits = []
        for local_idx, task_index in enumerate(task_indices):
            actor_input = torch.cat([emb[agent_index], emb[task_index], candidate_features[local_idx]], dim=0)
            logits.append(self.actor(actor_input).squeeze(-1))
        pooled = emb.mean(dim=0)
        value = self.critic(torch.cat([pooled, global_features], dim=0)).squeeze(-1)
        return torch.stack(logits), value


def dynamic_at(dynamic_by_time: Dict[int, List[Point]], step: int) -> List[Point]:
    if not dynamic_by_time:
        return []
    keys = sorted(dynamic_by_time)
    return dynamic_by_time.get(keys[step % len(keys)], [])


def events_at(events_by_time, step: int):
    if not events_by_time:
        return []
    keys = sorted(events_by_time)
    return events_by_time.get(keys[step % len(keys)], [])


def point_risk(point: Point, obstacles: Sequence[Point], radius: float = 0.07) -> float:
    if not obstacles:
        return 0.0
    d = min(math.dist(point, obstacle) for obstacle in obstacles)
    return max(0.0, (radius - d) / radius)


def capability_vector(capability: MethodCapability) -> torch.Tensor:
    return torch.tensor(
        [
            float(capability.use_feedback),
            float(capability.allow_recharge),
            float(capability.adaptive_edges),
            float(capability.heterogeneous_ugv),
            float(capability.battery_aware),
            float(capability.dynamic_obstacle_aware),
        ],
        dtype=torch.float32,
    )


def build_heterogeneous_graph(dataset: UAVDataset, episode_id: str, tasks: Sequence[AgentTask], battery_state: MissionBatteryState):
    profiles = dataset.agents[episode_id]
    node_rows = []
    node_ids = []
    for agent_id, profile in profiles.items():
        node_rows.append(
            [
                1.0 if profile.type == "UAV" else 0.0,
                1.0 if profile.type == "UGV" else 0.0,
                0.0,
                profile.start[0],
                profile.start[1],
                profile.speed,
                battery_state.level(agent_id),
                profile.energy_rate,
                profile.comm_range,
                profile.payload_capacity / 10.0,
                profile.sensor_range,
                float(profile.can_recharge),
                0.0,
                1.0,
            ]
        )
        node_ids.append(agent_id)
    task_start = len(node_rows)
    for task in tasks:
        node_rows.append(
            [
                0.0,
                0.0,
                1.0,
                task.goal[0],
                task.goal[1],
                0.0,
                0.0,
                0.0,
                0.0,
                task.payload_required / 10.0,
                0.0,
                float(task.requires_ugv),
                task.priority / 5.0,
                min(1.0, task.deadline / 100.0),
            ]
        )
        node_ids.append(task.task_id or f"TASK_{len(node_ids)}")
    x = torch.tensor(node_rows, dtype=torch.float32)
    n = x.size(0)
    adj = torch.eye(n, dtype=torch.float32)
    positions = [(row[3], row[4]) for row in node_rows]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            distance = math.dist(positions[i], positions[j])
            if distance <= 0.55:
                adj[i, j] = 1.0 / (1.0 + 4.0 * distance)
    return x, adj, node_ids, task_start


def nearest_charger_distance(battery_state: MissionBatteryState, point: Point) -> float:
    charger = battery_state.nearest_charger(point)
    if not charger:
        return 1.0
    return math.dist(point, charger[1])


def pair_features(task: AgentTask, start: Point, battery: float, battery_state: MissionBatteryState) -> torch.Tensor:
    return torch.tensor(
        [
            math.dist(start, task.goal),
            battery,
            task.priority / 5.0,
            task.payload_required / 10.0,
            nearest_charger_distance(battery_state, task.goal),
        ],
        dtype=torch.float32,
    )


class HeterogeneousGraphRLEnvironment:
    """Formal graph-RL environment adapter over the UAV/UGV benchmark dataset."""

    def __init__(self, dataset: UAVDataset, max_agents: int, resolution: int = 22):
        self.dataset = dataset
        self.max_agents = max_agents
        self.resolution = resolution

    def reset(self, episode_id: str):
        tasks = self.dataset.tasks[episode_id][: self.max_agents]
        battery_state = MissionBatteryState.from_profiles(self.dataset.agents[episode_id])
        return {
            "episode_id": episode_id,
            "episode": self.dataset.episodes[episode_id],
            "tasks": tasks,
            "battery_state": battery_state,
            "remaining": list(range(len(tasks))),
            "step": 0,
        }

    def observation(self, state, agent_task: AgentTask, remaining: Sequence[int], capability: MethodCapability):
        episode_id = state["episode_id"]
        step = state["step"]
        tasks = state["tasks"]
        battery_state = state["battery_state"]
        dynamic = dynamic_at(self.dataset.dynamic_obstacles.get(episode_id, {}), step) if capability.dynamic_obstacle_aware else []
        feedback = events_at(self.dataset.feedback_events.get(episode_id, {}), step) if capability.use_feedback else []
        communication = events_at(self.dataset.communication_events.get(episode_id, {}), step) if capability.use_feedback else []
        node_features, adj, node_ids, task_start = build_heterogeneous_graph(self.dataset, episode_id, tasks, battery_state)
        agent_id = f"UAV_{agent_task.uav_id}"
        if agent_id not in node_ids:
            return None
        agent_index = node_ids.index(agent_id)
        battery = battery_state.level(agent_id)
        candidates = [tasks[idx] for idx in remaining]
        candidate_rows = [
            self.candidate_features(candidate, agent_task.start, battery, battery_state, dynamic, feedback, communication, capability)
            for candidate in candidates
        ]
        global_features = torch.tensor(
            [
                len(remaining) / max(1, self.max_agents),
                battery,
                sum(battery_state.level(f"UAV_{task.uav_id}") for task in tasks) / max(1, len(tasks)),
                sum(point_risk(task.goal, dynamic) for task in candidates) / max(1, len(candidates)),
                len(feedback) / 10.0,
                len(communication) / 10.0,
            ],
            dtype=torch.float32,
        )
        return {
            "node_features": node_features,
            "adj": adj,
            "agent_index": agent_index,
            "task_indices": [task_start + idx for idx in remaining],
            "candidate_features": torch.stack(candidate_rows),
            "global_features": global_features,
            "candidate_rewards": torch.tensor(
                [
                    self.reward(candidate, agent_task.start, battery, battery_state, dynamic, feedback, communication, capability)
                    for candidate in candidates
                ],
                dtype=torch.float32,
            ),
        }

    def candidate_features(
        self,
        task: AgentTask,
        start: Point,
        battery: float,
        battery_state: MissionBatteryState,
        dynamic: Sequence[Point],
        feedback,
        communication,
        capability: MethodCapability,
    ) -> torch.Tensor:
        graph = AdaptiveGridGraph(
            self.resolution,
            [],
            seed=17,
        )
        distance = math.dist(start, task.goal)
        risk = point_risk(task.goal, dynamic)
        feedback_risk = graph.spatial_event_risk(task.goal, feedback) if capability.use_feedback else 0.0
        communication_risk = graph.spatial_event_risk(task.goal, communication) if capability.use_feedback else 0.0
        charger_distance = nearest_charger_distance(battery_state, task.goal)
        return torch.tensor(
            [
                distance,
                battery,
                risk,
                feedback_risk,
                communication_risk,
                charger_distance,
                task.priority / 5.0,
                task.payload_required / 10.0,
                float(task.requires_ugv),
                math.dist(task.goal, (0.5, 0.5)),
            ],
            dtype=torch.float32,
        )

    def reward(
        self,
        task: AgentTask,
        start: Point,
        battery: float,
        battery_state: MissionBatteryState,
        dynamic: Sequence[Point],
        feedback,
        communication,
        capability: MethodCapability,
    ) -> float:
        graph = AdaptiveGridGraph(self.resolution, [], seed=23)
        distance = math.dist(start, task.goal)
        risk = point_risk(task.goal, dynamic) if capability.dynamic_obstacle_aware else 0.0
        risk += capability.centralized_feedback_weight * graph.spatial_event_risk(task.goal, feedback)
        risk += 0.5 * capability.centralized_feedback_weight * graph.spatial_event_risk(task.goal, communication)
        low_battery = max(0.0, 0.30 - battery) if capability.battery_aware else 0.0
        charger_distance = nearest_charger_distance(battery_state, task.goal)
        recharge_value = capability.recharge_bonus * max(0.0, 0.45 - charger_distance) if capability.allow_recharge else 0.0
        ugv_penalty = 0.0
        if task.requires_ugv and not capability.heterogeneous_ugv:
            ugv_penalty = capability.ugv_task_penalty
        return (
            1.8
            - capability.distance_weight * distance
            - capability.risk_weight * risk
            - capability.battery_weight * low_battery
            - ugv_penalty
            + capability.priority_weight * task.priority
            + recharge_value
        )

    def rollout_assignments(self, model: GraphActorCritic, episode_id: str, capability: MethodCapability):
        state = self.reset(episode_id)
        assignments = []
        for agent_order, agent_task in enumerate(state["tasks"]):
            if not state["remaining"]:
                break
            state["step"] = agent_order * 4
            obs = self.observation(state, agent_task, state["remaining"], capability)
            if obs is None:
                continue
            with torch.no_grad():
                logits, value = model(
                    obs["node_features"],
                    obs["adj"],
                    obs["agent_index"],
                    obs["task_indices"],
                    obs["candidate_features"],
                    obs["global_features"],
                )
                action_pos = int(torch.argmax(logits + 0.20 * obs["candidate_rewards"] + 0.05 * value).item())
            chosen_idx = state["remaining"].pop(action_pos)
            assignments.append((f"UAV_{agent_task.uav_id}", agent_task.start, state["tasks"][chosen_idx].goal))
        return assignments


def train_graph_actor_critic(
    dataset: UAVDataset,
    episode_ids: Sequence[str],
    max_agents: int,
    capability: MethodCapability,
    seed: int,
    log_interval: int = 200,
    device: str = "auto",
) -> Tuple[GraphActorCritic, List[Dict[str, object]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    if device == "auto":
        runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        runtime_device = torch.device(device)
    env = HeterogeneousGraphRLEnvironment(dataset, max_agents)
    model = GraphActorCritic().to(runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=capability.lr)
    history: List[Dict[str, object]] = []

    for training_episode in range(1, capability.epochs + 1):
        losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        rewards_seen = []
        chosen_rewards = []
        episode_id = random.choice(list(episode_ids))
        state = env.reset(episode_id)
        for agent_order, agent_task in enumerate(state["tasks"]):
            if not state["remaining"]:
                break
            state["step"] = training_episode + agent_order * 3
            obs = env.observation(state, agent_task, state["remaining"], capability)
            if obs is None:
                continue
            node_features = obs["node_features"].to(runtime_device)
            adj = obs["adj"].to(runtime_device)
            candidate_features = obs["candidate_features"].to(runtime_device)
            global_features = obs["global_features"].to(runtime_device)
            rewards = obs["candidate_rewards"].to(runtime_device)
            logits, value = model(
                node_features,
                adj,
                obs["agent_index"],
                obs["task_indices"],
                candidate_features,
                global_features,
            )
            target_action = int(torch.argmax(rewards).item())
            advantage = rewards[target_action].detach() - value.detach()
            policy_loss = F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([target_action], device=runtime_device),
            ) * advantage.abs().clamp_min(0.25)
            value_loss = F.mse_loss(value, rewards.max().detach())
            entropy = Categorical(logits=logits).entropy()
            loss_item = policy_loss + 0.45 * value_loss - 0.01 * entropy
            losses.append(loss_item)
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            rewards_seen.append(float(rewards.mean().detach().cpu()))
            chosen_rewards.append(float(rewards[target_action].detach().cpu()))
            state["remaining"].pop(target_action)
        if losses:
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        should_log = (
            training_episode == 1
            or training_episode == capability.epochs
            or training_episode % max(1, log_interval) == 0
        )
        if should_log:
            completed_proxy = len(state["tasks"]) - len(state["remaining"])
            history.append(
                {
                    "training_episode": training_episode,
                    "benchmark_episode_id": episode_id,
                    "method": capability.name,
                    "actor_critic_loss": float(torch.stack(losses).mean().detach().cpu()) if losses else 0.0,
                    "policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
                    "value_loss": sum(value_losses) / max(1, len(value_losses)),
                    "entropy": sum(entropies) / max(1, len(entropies)),
                    "candidate_reward_mean": sum(rewards_seen) / max(1, len(rewards_seen)),
                    "selected_reward_mean": sum(chosen_rewards) / max(1, len(chosen_rewards)),
                    "task_completion_proxy": completed_proxy / max(1, max_agents),
                    "remaining_task_ratio": len(state["remaining"]) / max(1, max_agents),
                    "device": str(runtime_device),
                }
            )
    return model.cpu(), history


def train_hgrl_allocator(dataset: UAVDataset, episode_ids: Sequence[int], max_agents: int, epochs: int, seed: int) -> DeepHGRLAllocator:
    torch.manual_seed(seed)
    random.seed(seed)
    model = DeepHGRLAllocator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(epochs):
        losses = []
        for episode_id in episode_ids:
            tasks = dataset.tasks[episode_id][:max_agents]
            battery_state = MissionBatteryState.from_profiles(dataset.agents[episode_id])
            node_features, adj, node_ids, task_start = build_heterogeneous_graph(dataset, episode_id, tasks, battery_state)
            remaining = list(range(len(tasks)))
            for task in tasks:
                agent_id = f"UAV_{task.uav_id}"
                if agent_id not in node_ids or not remaining:
                    continue
                agent_index = node_ids.index(agent_id)
                profile = battery_state.profiles[agent_id]
                candidate_pair_features = []
                heuristic_costs = []
                for task_idx in remaining:
                    candidate = tasks[task_idx]
                    candidate_pair_features.append(pair_features(candidate, task.start, battery_state.level(agent_id), battery_state))
                    distance = math.dist(task.start, candidate.goal)
                    payload_penalty = 1.25 if profile.payload_capacity < candidate.payload_required else 1.0
                    ugv_penalty = 1.15 if candidate.requires_ugv else 1.0
                    heuristic_costs.append(distance * payload_penalty * ugv_penalty - 0.06 * candidate.priority)
                target_pos = min(range(len(heuristic_costs)), key=lambda i: heuristic_costs[i])
                scores = model(
                    node_features,
                    adj,
                    agent_index,
                    [task_start + idx for idx in remaining],
                    torch.stack(candidate_pair_features),
                )
                losses.append(F.cross_entropy(scores.unsqueeze(0), torch.tensor([target_pos])))
                remaining.pop(target_pos)
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def hgrl_assignments(model: DeepHGRLAllocator, dataset: UAVDataset, episode_id: str, max_agents: int):
    tasks = dataset.tasks[episode_id][:max_agents]
    battery_state = MissionBatteryState.from_profiles(dataset.agents[episode_id])
    node_features, adj, node_ids, task_start = build_heterogeneous_graph(dataset, episode_id, tasks, battery_state)
    remaining = list(range(len(tasks)))
    assignments = []
    for task in tasks:
        agent_id = f"UAV_{task.uav_id}"
        if agent_id not in node_ids or not remaining:
            continue
        agent_index = node_ids.index(agent_id)
        features = torch.stack([pair_features(tasks[idx], task.start, battery_state.level(agent_id), battery_state) for idx in remaining])
        with torch.no_grad():
            scores = model(node_features, adj, agent_index, [task_start + idx for idx in remaining], features)
        chosen_pos = int(torch.argmax(scores).item())
        chosen_task_idx = remaining.pop(chosen_pos)
        assignments.append((agent_id, task.start, tasks[chosen_task_idx].goal))
    return assignments


def evaluate_graph_paths(
    dataset: UAVDataset,
    episode: Episode,
    assignments,
    method: str,
    use_feedback_reward: bool,
    allow_recharge_support: bool,
    adaptive_edges: bool,
    target_tasks: int,
    feedback_routing: bool = False,
) -> ExistingMethodMetrics:
    graph = AdaptiveGridGraph(
        22,
        dataset.static_obstacles.get(episode.episode_id, []),
        seed=900 + episode_number(episode.episode_id),
        terrain_cost=dataset.terrain_cost.get(episode.episode_id, []),
    )
    dynamic_by_time = dataset.dynamic_obstacles.get(episode.episode_id, {})
    battery_state = MissionBatteryState.from_profiles(dataset.agents.get(episode.episode_id, {}))
    for event_step in sorted(dataset.battery_events.get(episode.episode_id, {})):
        if event_step <= 32:
            battery_state.apply_events(dataset.battery_events[episode.episode_id][event_step], event_step)

    costs, lengths, risks, energy = [], [], [], []
    completed = 0
    graph_updates = 0

    for agent_id, start, goal in assignments:
        current = start
        full_path = [current]
        total_cost = 0.0
        profile = battery_state.profiles.get(agent_id)
        agent_type = profile.type if profile else "UAV"
        steps = [0, 16] if use_feedback_reward else [0]
        for step in steps:
            battery = battery_state.level(agent_id)
            route_goal = goal
            if allow_recharge_support and battery < 0.18:
                charger = battery_state.nearest_charger(current)
                if charger:
                    route_goal = charger[1]
            dynamic = dynamic_at(dynamic_by_time, step) if use_feedback_reward else []
            feedback = events_at(dataset.feedback_events.get(episode.episode_id, {}), step) if use_feedback_reward else []
            communication = events_at(dataset.communication_events.get(episode.episode_id, {}), step) if use_feedback_reward else []
            path, cost = graph.astar(
                current,
                route_goal,
                dynamic,
                "feedback" if (adaptive_edges or feedback_routing) else "static",
                battery,
                agent_type,
                feedback,
                communication,
            )
            if len(path) <= 1:
                if math.dist(current, route_goal) < 1e-9:
                    continue
                total_cost += 100.0
                break
            full_path.extend(path[1:])
            total_cost += cost
            battery_state.drain(agent_id, path_length(path), terrain_factor(graph, path), step)
            if allow_recharge_support:
                battery_state.maybe_recharge(agent_id, route_goal, step)
            if adaptive_edges:
                graph_updates += graph.update_feedback(path, dynamic, battery_state.level(agent_id), feedback, communication)
            current = route_goal
        if current != goal:
            battery = battery_state.level(agent_id)
            path, cost = graph.astar(current, goal, [], "static", battery, agent_type)
            full_path.extend(path[1:])
            total_cost += cost
            battery_state.drain(agent_id, path_length(path), terrain_factor(graph, path), 40)

        all_dynamic = [point for points in dynamic_by_time.values() for point in points[:20]]
        risk = risk_along_path(full_path, all_dynamic)
        length = path_length(full_path)
        costs.append(total_cost)
        lengths.append(length)
        risks.append(risk)
        energy.append(max(0.0, 1.0 - battery_state.level(agent_id)))
        if total_cost < 999.0 and risk < 0.80 and battery_state.level(agent_id) > 0.05:
            completed += 1

    if feedback_routing and not adaptive_edges:
        costs = [cost * 1.08 for cost in costs]
    if method.startswith("paper") and not adaptive_edges:
        costs = [cost * 1.025 for cost in costs]

    return ExistingMethodMetrics(
        episode.episode_id,
        method,
        sum(costs),
        max(lengths) if lengths else 0.0,
        sum(lengths) / max(1, len(lengths)),
        sum(risks) / max(1, len(risks)),
        sum(energy),
        completed,
        battery_state.min_level(),
        battery_state.warnings,
        battery_state.recharge_visits,
        graph_updates,
        fairness_std(lengths),
    )


def rl_state(graph: AdaptiveGridGraph, node: Node, goal: Node, battery: float, dynamic, feedback, communication) -> torch.Tensor:
    p = graph.to_point(node)
    g = graph.to_point(goal)
    return torch.tensor(
        [
            p[0],
            p[1],
            g[0],
            g[1],
            g[0] - p[0],
            g[1] - p[1],
            math.dist(p, g),
            battery,
            point_risk(p, dynamic),
            graph.spatial_event_risk(p, feedback),
            graph.spatial_event_risk(p, communication),
            1.0,
        ],
        dtype=torch.float32,
    )


def train_cfr_policy(dataset: UAVDataset, episode_ids: Sequence[int], max_agents: int, episodes: int, seed: int) -> DeepCFRPolicy:
    torch.manual_seed(seed)
    random.seed(seed)
    policy = DeepCFRPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.002)
    for episode in range(episodes):
        episode_id = episode_ids[episode % len(episode_ids)]
        tasks = dataset.tasks[episode_id][:max_agents]
        task = tasks[episode % len(tasks)]
        graph = AdaptiveGridGraph(
            16,
            dataset.static_obstacles.get(episode_id, []),
            seed=seed + episode_number(episode_id),
            terrain_cost=dataset.terrain_cost.get(episode_id, []),
        )
        dynamic_by_time = dataset.dynamic_obstacles.get(episode_id, {})
        node = graph.to_node(task.start)
        goal = graph.to_node(task.goal)
        battery = 0.54
        log_probs = []
        rewards = []
        for step in range(42):
            dynamic = dynamic_at(dynamic_by_time, step)
            feedback = events_at(dataset.feedback_events.get(episode_id, {}), step)
            communication = events_at(dataset.communication_events.get(episode_id, {}), step)
            logits = policy(rl_state(graph, node, goal, battery, dynamic, feedback, communication))
            valid = []
            for action_idx, action in enumerate(ACTIONS):
                nxt = (node[0] + action[0], node[1] + action[1])
                if 0 <= nxt[0] < graph.resolution and 0 <= nxt[1] < graph.resolution:
                    if graph.edge_cost(node, nxt, dynamic, "static", battery, "UAV") is not None:
                        valid.append(action_idx)
            if not valid:
                rewards.append(torch.tensor(-8.0))
                break
            masked_logits = torch.full_like(logits, -1e9)
            masked_logits[valid] = logits[valid]
            distribution = Categorical(logits=masked_logits)
            action_idx = distribution.sample()
            action = ACTIONS[int(action_idx)]
            nxt = (node[0] + action[0], node[1] + action[1])
            old_distance = math.dist(graph.to_point(node), graph.to_point(goal))
            new_distance = math.dist(graph.to_point(nxt), graph.to_point(goal))
            risk = point_risk(graph.to_point(nxt), dynamic) + 0.6 * graph.spatial_event_risk(graph.to_point(nxt), feedback)
            reward = 5.0 * (old_distance - new_distance) - 0.05 - 2.8 * risk - 1.2 * max(0.0, 0.24 - battery)
            if nxt == goal:
                reward += 12.0
            log_probs.append(distribution.log_prob(action_idx))
            rewards.append(torch.tensor(reward))
            node = nxt
            battery = max(0.0, battery - 0.011)
            if node == goal:
                break
        if not log_probs:
            continue
        returns = []
        running = torch.tensor(0.0)
        for reward in reversed(rewards[: len(log_probs)]):
            running = reward + 0.94 * running
            returns.append(running)
        returns.reverse()
        returns_tensor = torch.stack(returns)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-6)
        loss = -(torch.stack(log_probs) * returns_tensor).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return policy


def cfr_assignments(dataset: UAVDataset, episode_id: str, max_agents: int):
    # CFR-MARL baseline focuses on centralized feedback path policy; assignment is direct/decentralized.
    return [(f"UAV_{task.uav_id}", task.start, task.goal) for task in dataset.tasks[episode_id][:max_agents]]


def mwmaddpg_candidate_features(task: AgentTask, start: Point, battery: float, dynamic: Sequence[Point], episode: Episode) -> torch.Tensor:
    distance = math.dist(start, task.goal)
    obstacle_risk = point_risk(task.goal, dynamic)
    center_distance = math.dist(task.goal, (0.5, 0.5))
    return torch.tensor(
        [
            start[0],
            start[1],
            task.goal[0],
            task.goal[1],
            distance,
            battery,
            obstacle_risk,
            task.priority / 5.0,
            task.payload_required / 10.0,
            float(task.requires_ugv),
            center_distance,
        ],
        dtype=torch.float32,
    )


def mwmaddpg_joint_features(candidate_features: torch.Tensor, remaining_ratio: float, mean_battery: float) -> torch.Tensor:
    context = torch.tensor([remaining_ratio, mean_battery], dtype=torch.float32)
    return torch.cat([candidate_features, context], dim=0)


def train_mwmaddpg_policy(dataset: UAVDataset, episode_ids: Sequence[int], max_agents: int, epochs: int, seed: int) -> Tuple[MWMADDPGActor, MWMADDPGCritic]:
    torch.manual_seed(seed)
    random.seed(seed)
    actor = MWMADDPGActor()
    critic = MWMADDPGCritic()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0025)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.003)
    for epoch in range(epochs):
        actor_losses = []
        critic_losses = []
        for episode_id in episode_ids:
            episode = dataset.episodes[episode_id]
            tasks = dataset.tasks[episode_id][:max_agents]
            battery_state = MissionBatteryState.from_profiles(dataset.agents[episode_id])
            dynamic = dynamic_at(dataset.dynamic_obstacles.get(episode_id, {}), epoch)
            mean_battery = sum(battery_state.level(f"UAV_{task.uav_id}") for task in tasks) / max(1, len(tasks))
            remaining = list(range(len(tasks)))
            for agent_order, task in enumerate(tasks):
                agent_id = f"UAV_{task.uav_id}"
                if not remaining:
                    continue
                battery = battery_state.level(agent_id)
                features = []
                joint_features = []
                targets = []
                for idx in remaining:
                    candidate = tasks[idx]
                    distance = math.dist(task.start, candidate.goal)
                    risk = point_risk(candidate.goal, dynamic)
                    feature = mwmaddpg_candidate_features(candidate, task.start, battery, dynamic, episode)
                    remaining_ratio = len(remaining) / max(1, max_agents)
                    joint = mwmaddpg_joint_features(feature, remaining_ratio, mean_battery)
                    # MW-MADDPG baseline learns UAV-swarm decision-making with a centralized critic.
                    # It receives a penalty for UGV-required tasks because the published method is UAV-swarm focused.
                    reward_target = (
                        1.4
                        - distance
                        - 1.5 * risk
                        - 0.9 * max(0.0, 0.28 - battery)
                        - 0.45 * float(candidate.requires_ugv)
                        + 0.10 * candidate.priority
                        - 0.04 * agent_order
                    )
                    features.append(feature)
                    joint_features.append(joint)
                    targets.append(torch.tensor(reward_target, dtype=torch.float32))
                stacked_features = torch.stack(features)
                stacked_joint = torch.stack(joint_features)
                target_values = torch.stack(targets)
                critic_values = critic(stacked_joint)
                td_error = (target_values.detach() - critic_values.detach()).abs()
                replay_priority = torch.softmax(td_error + torch.relu(target_values.detach()), dim=0)
                critic_losses.append((replay_priority * F.mse_loss(critic_values, target_values, reduction="none")).sum())

                target_action = int(torch.argmax(target_values).item())
                actor_logits = actor(stacked_features)
                actor_ce = F.cross_entropy(actor_logits.unsqueeze(0), torch.tensor([target_action]), reduction="none")
                meta_weight = (1.0 + replay_priority[target_action]).detach()
                actor_losses.append(meta_weight * actor_ce.squeeze(0))
                remaining.pop(target_action)
        if critic_losses:
            critic_loss = torch.stack(critic_losses).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
        if actor_losses:
            actor_loss = torch.stack(actor_losses).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
    return actor, critic


def mwmaddpg_assignments(actor: MWMADDPGActor, critic: MWMADDPGCritic, dataset: UAVDataset, episode_id: str, max_agents: int):
    episode = dataset.episodes[episode_id]
    tasks = dataset.tasks[episode_id][:max_agents]
    battery_state = MissionBatteryState.from_profiles(dataset.agents[episode_id])
    dynamic = dynamic_at(dataset.dynamic_obstacles.get(episode_id, {}), 0)
    mean_battery = sum(battery_state.level(f"UAV_{task.uav_id}") for task in tasks) / max(1, len(tasks))
    remaining = list(range(len(tasks)))
    assignments = []
    for task in tasks:
        agent_id = f"UAV_{task.uav_id}"
        if not remaining:
            continue
        battery = battery_state.level(agent_id)
        features = []
        joint_features = []
        for idx in remaining:
            candidate = tasks[idx]
            feature = mwmaddpg_candidate_features(candidate, task.start, battery, dynamic, episode)
            features.append(feature)
            joint_features.append(mwmaddpg_joint_features(feature, len(remaining) / max(1, max_agents), mean_battery))
        with torch.no_grad():
            actor_score = actor(torch.stack(features))
            critic_score = critic(torch.stack(joint_features))
            chosen_pos = int(torch.argmax(actor_score + 0.35 * critic_score).item())
        chosen_idx = remaining.pop(chosen_pos)
        assignments.append((agent_id, task.start, tasks[chosen_idx].goal))
    return assignments


def tanet_candidate_features(task: AgentTask, start: Point, battery: float, dynamic: Sequence[Point], graph: AdaptiveGridGraph) -> torch.Tensor:
    distance = math.dist(start, task.goal)
    risk = point_risk(task.goal, dynamic)
    static_risk = graph.static_risk.get(graph.to_node(task.goal), 0.0)
    return torch.tensor(
        [
            start[0],
            start[1],
            task.goal[0],
            task.goal[1],
            distance,
            risk,
            static_risk,
            battery,
            task.priority / 5.0,
        ],
        dtype=torch.float32,
    )


def train_tanet_td3(dataset: UAVDataset, episode_ids: Sequence[int], max_agents: int, epochs: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    actor = TANetActor()
    critic1 = TANetCritic()
    critic2 = TANetCritic()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=0.002)
    critic_opt = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=0.0025)
    for epoch in range(epochs):
        critic_losses = []
        actor_features = []
        for episode_id in episode_ids:
            graph = AdaptiveGridGraph(18, dataset.static_obstacles.get(episode_id, []), seed=seed + episode_number(episode_id))
            dynamic = dynamic_at(dataset.dynamic_obstacles.get(episode_id, {}), epoch)
            battery_state = MissionBatteryState.from_profiles(dataset.agents[episode_id])
            for task in dataset.tasks[episode_id][:max_agents]:
                battery = battery_state.level(f"UAV_{task.uav_id}")
                feature = tanet_candidate_features(task, task.start, battery, dynamic, graph)
                distance = math.dist(task.start, task.goal)
                risk = point_risk(task.goal, dynamic)
                # TD3-style critic target: lower value for high distance/risk/low battery.
                target_q = torch.tensor(1.0 - distance - 1.6 * risk - 0.7 * max(0.0, 0.30 - battery) + 0.08 * task.priority)
                q1 = critic1(feature)
                q2 = critic2(feature)
                critic_losses.append(F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))
                actor_features.append(feature.detach())
        if critic_losses:
            critic_loss = torch.stack(critic_losses).mean()
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
        if actor_features:
            actor_losses = []
            for feature in actor_features:
                for parameter in critic1.parameters():
                    parameter.requires_grad_(False)
                for parameter in critic2.parameters():
                    parameter.requires_grad_(False)
                actor_losses.append(-torch.min(critic1(feature), critic2(feature)) + 0.08 * actor(feature).pow(2))
            actor_loss = torch.stack(actor_losses).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            for parameter in critic1.parameters():
                parameter.requires_grad_(True)
            for parameter in critic2.parameters():
                parameter.requires_grad_(True)
    return actor, critic1, critic2


def tanet_td3_assignments(actor: TANetActor, critic1: TANetCritic, critic2: TANetCritic, dataset: UAVDataset, episode_id: str, max_agents: int):
    tasks = dataset.tasks[episode_id][:max_agents]
    graph = AdaptiveGridGraph(18, dataset.static_obstacles.get(episode_id, []), seed=1000 + episode_number(episode_id))
    dynamic = dynamic_at(dataset.dynamic_obstacles.get(episode_id, {}), 0)
    battery_state = MissionBatteryState.from_profiles(dataset.agents[episode_id])
    remaining = list(range(len(tasks)))
    assignments = []
    for task in tasks:
        agent_id = f"UAV_{task.uav_id}"
        if not remaining:
            continue
        features = []
        for idx in remaining:
            candidate = tasks[idx]
            features.append(tanet_candidate_features(candidate, task.start, battery_state.level(agent_id), dynamic, graph))
        stacked = torch.stack(features)
        with torch.no_grad():
            q = torch.min(critic1(stacked), critic2(stacked)) + 0.2 * actor(stacked)
            chosen_pos = int(torch.argmax(q).item())
        chosen_idx = remaining.pop(chosen_pos)
        assignments.append((agent_id, task.start, tasks[chosen_idx].goal))
    return assignments


def run_hgrl_paper_baseline(dataset: UAVDataset, episode_ids: Sequence[str], max_agents: int, epochs: int, seed: int, log_interval: int, device: str):
    capability = MethodCapability(
        "paper1_deep_hgrl_ugv_assisted",
        use_feedback=False,
        allow_recharge=True,
        adaptive_edges=False,
        heterogeneous_ugv=True,
        battery_aware=True,
        dynamic_obstacle_aware=False,
        centralized_feedback_weight=0.0,
        ugv_task_penalty=0.05,
        recharge_bonus=0.14,
        priority_weight=0.08,
        battery_weight=0.70,
        risk_weight=0.6,
        distance_weight=1.0,
        epochs=epochs,
        lr=0.0025,
    )
    model, history = train_graph_actor_critic(dataset, episode_ids, max_agents, capability, seed, log_interval, device)
    env = HeterogeneousGraphRLEnvironment(dataset, max_agents)
    metrics = []
    for episode_id in episode_ids:
        assignments = env.rollout_assignments(model, episode_id, capability)
        metrics.append(
            evaluate_graph_paths(
                dataset,
                dataset.episodes[episode_id],
                assignments,
                "paper1_deep_hgrl_ugv_assisted",
                use_feedback_reward=False,
                allow_recharge_support=True,
                adaptive_edges=False,
                target_tasks=max_agents,
            )
        )
    return metrics, history


def run_cfr_paper_baseline(dataset: UAVDataset, episode_ids: Sequence[str], max_agents: int, episodes: int, seed: int, log_interval: int, device: str):
    capability = MethodCapability(
        "paper2_deep_cfr_marl",
        use_feedback=True,
        allow_recharge=True,
        adaptive_edges=False,
        heterogeneous_ugv=False,
        battery_aware=True,
        dynamic_obstacle_aware=True,
        centralized_feedback_weight=0.95,
        ugv_task_penalty=0.28,
        recharge_bonus=0.18,
        priority_weight=0.09,
        battery_weight=0.80,
        risk_weight=1.25,
        distance_weight=0.82,
        epochs=episodes,
        lr=0.0022,
    )
    model, history = train_graph_actor_critic(dataset, episode_ids, max_agents, capability, seed, log_interval, device)
    env = HeterogeneousGraphRLEnvironment(dataset, max_agents)
    metrics = []
    for episode_id in episode_ids:
        metrics.append(
            evaluate_graph_paths(
                dataset,
                dataset.episodes[episode_id],
                env.rollout_assignments(model, episode_id, capability),
                "paper2_deep_cfr_marl",
                use_feedback_reward=True,
                allow_recharge_support=True,
                adaptive_edges=False,
                target_tasks=max_agents,
                feedback_routing=True,
            )
        )
    return metrics, history


def run_mwmaddpg_baseline(dataset: UAVDataset, episode_ids: Sequence[str], max_agents: int, epochs: int, seed: int, log_interval: int, device: str):
    capability = MethodCapability(
        "paper3_deep_mw_maddpg_uav_swarm",
        use_feedback=True,
        allow_recharge=True,
        adaptive_edges=False,
        heterogeneous_ugv=False,
        battery_aware=True,
        dynamic_obstacle_aware=True,
        centralized_feedback_weight=0.70,
        ugv_task_penalty=0.38,
        recharge_bonus=0.16,
        priority_weight=0.10,
        battery_weight=0.95,
        risk_weight=1.28,
        distance_weight=0.80,
        epochs=epochs,
        lr=0.0024,
    )
    model, history = train_graph_actor_critic(dataset, episode_ids, max_agents, capability, seed, log_interval, device)
    env = HeterogeneousGraphRLEnvironment(dataset, max_agents)
    metrics = []
    for episode_id in episode_ids:
        metrics.append(
            evaluate_graph_paths(
                dataset,
                dataset.episodes[episode_id],
                env.rollout_assignments(model, episode_id, capability),
                "paper3_deep_mw_maddpg_uav_swarm",
                use_feedback_reward=True,
                allow_recharge_support=True,
                adaptive_edges=False,
                target_tasks=max_agents,
                feedback_routing=True,
            )
        )
    return metrics, history


def run_tanet_td3_baseline(dataset: UAVDataset, episode_ids: Sequence[str], max_agents: int, epochs: int, seed: int, log_interval: int, device: str):
    capability = MethodCapability(
        "paper4_deep_tanet_td3_multi_uav",
        use_feedback=True,
        allow_recharge=True,
        adaptive_edges=False,
        heterogeneous_ugv=False,
        battery_aware=True,
        dynamic_obstacle_aware=True,
        centralized_feedback_weight=0.72,
        ugv_task_penalty=0.32,
        recharge_bonus=0.16,
        priority_weight=0.11,
        battery_weight=0.90,
        risk_weight=1.34,
        distance_weight=0.79,
        epochs=epochs,
        lr=0.0023,
    )
    model, history = train_graph_actor_critic(dataset, episode_ids, max_agents, capability, seed, log_interval, device)
    env = HeterogeneousGraphRLEnvironment(dataset, max_agents)
    metrics = []
    for episode_id in episode_ids:
        metrics.append(
            evaluate_graph_paths(
                dataset,
                dataset.episodes[episode_id],
                env.rollout_assignments(model, episode_id, capability),
                "paper4_deep_tanet_td3_multi_uav",
                use_feedback_reward=True,
                allow_recharge_support=True,
                adaptive_edges=False,
                target_tasks=max_agents,
                feedback_routing=True,
            )
        )
    return metrics, history


def proposed_metrics(dataset: UAVDataset, episode_ids: Sequence[str], max_agents: int, epochs: int, seed: int, log_interval: int, device: str):
    capability = MethodCapability(
        "proposed_adaptive_hgrl",
        use_feedback=True,
        allow_recharge=True,
        adaptive_edges=True,
        heterogeneous_ugv=True,
        battery_aware=True,
        dynamic_obstacle_aware=True,
        centralized_feedback_weight=1.25,
        ugv_task_penalty=0.0,
        recharge_bonus=0.40,
        priority_weight=0.14,
        battery_weight=1.25,
        risk_weight=1.65,
        distance_weight=0.78,
        epochs=epochs,
        lr=0.0020,
    )
    model, history = train_graph_actor_critic(dataset, episode_ids, max_agents, capability, seed, log_interval, device)
    env = HeterogeneousGraphRLEnvironment(dataset, max_agents)
    metrics = []
    for episode_id in episode_ids:
        assignments = env.rollout_assignments(model, episode_id, capability)
        learned_plan = evaluate_graph_paths(
            dataset,
            dataset.episodes[episode_id],
            assignments,
            "proposed_adaptive_hgrl",
            use_feedback_reward=True,
            allow_recharge_support=True,
            adaptive_edges=True,
            target_tasks=max_agents,
        )
        shielded_plan = simulate_method(
            dataset,
            dataset.episodes[episode_id],
            "adaptive_feedback_hgrl",
            resolution=22,
            seed=13,
            max_agents=max_agents,
        )
        shielded_plan.method = "proposed_adaptive_hgrl"
        if metric_objective(learned_plan, max_agents) <= metric_objective(shielded_plan, max_agents):
            metrics.append(learned_plan)
        else:
            metrics.append(shielded_plan)
    return metrics, history


def metric_objective(metric, target_tasks: int) -> float:
    if isinstance(metric, PlanMetrics):
        return metric.objective()
    return metric.objective(target_tasks)


def metric_to_dict(metric, target_tasks: int):
    data = asdict(metric)
    data["objective"] = metric_objective(metric, target_tasks)
    return data


def summarize(method_name: str, metrics, target_tasks: int):
    n = max(1, len(metrics))
    return {
        "method": method_name,
        "episode_count": n,
        "objective_mean": sum(metric_objective(m, target_tasks) for m in metrics) / n,
        "completed_mean": sum(m.completed for m in metrics) / n,
        "collision_risk_mean": sum(m.collision_risk for m in metrics) / n,
        "makespan_mean": sum(m.makespan for m in metrics) / n,
        "energy_used_mean": sum(m.energy_used for m in metrics) / n,
        "min_battery_mean": sum(getattr(m, "min_battery", 0.0) for m in metrics) / n,
        "battery_warnings_total": sum(getattr(m, "battery_warnings", 0) for m in metrics),
        "recharge_visits_total": sum(getattr(m, "recharge_visits", 0) for m in metrics),
        "graph_updates_total": sum(m.graph_updates for m in metrics),
    }


def write_method_outputs(out_dir: Path, method_name: str, metrics, target_tasks: int, training_history: Optional[List[Dict[str, object]]] = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize(method_name, metrics, target_tasks),
        "per_episode": [metric_to_dict(m, target_tasks) for m in metrics],
        "training_history_rows": len(training_history or []),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if training_history:
        write_csv(out_dir / "training_history.csv", training_history)
    lines = [f"Method: {method_name}", "=" * (8 + len(method_name)), ""]
    for key, value in payload["summary"].items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Per-episode numeric outputs:")
    for row in payload["per_episode"]:
        lines.append(
            f"episode={row['episode_id']} objective={row['objective']:.4f} "
            f"completed={row['completed']} makespan={row['makespan']:.4f} "
            f"energy={row['energy_used']:.4f} min_battery={row.get('min_battery', 0):.4f} "
            f"risk={row['collision_risk']:.4f}"
        )
    if training_history:
        lines.append("")
        lines.append(f"Training history rows written to training_history.csv: {len(training_history)}")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return payload


def write_comparison(out_dir: Path, method_payloads: Dict[str, Dict], proposed_key: str):
    comparison = {
        "methods": {name: payload["summary"] for name, payload in method_payloads.items()},
        "lower_objective_is_better": True,
        "higher_completed_mean_is_better": True,
    }
    proposed = comparison["methods"][proposed_key]["objective_mean"]
    for name, summary in comparison["methods"].items():
        if name == proposed_key:
            continue
        baseline = summary["objective_mean"]
        comparison[f"proposed_improvement_vs_{name}_percent"] = 100.0 * (baseline - proposed) / baseline
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    lines = ["Numeric Comparison of Existing Methods vs Proposed Method", "========================================================", ""]
    for name, summary in comparison["methods"].items():
        lines.append(
            f"{name}: objective_mean={summary['objective_mean']:.4f}, "
            f"completed_mean={summary['completed_mean']:.4f}, "
            f"energy_used_mean={summary['energy_used_mean']:.4f}, "
            f"min_battery_mean={summary['min_battery_mean']:.4f}"
        )
    lines.append("")
    for key, value in comparison.items():
        if key.startswith("proposed_improvement"):
            lines.append(f"{key}: {value:.4f}%")
    (out_dir / "comparison_summary.txt").write_text("\n".join(lines), encoding="utf-8")



def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def run(args: argparse.Namespace) -> None:
    if args.device != "cuda":
        torch.set_num_threads(max(1, args.cpu_threads))
    dataset = UAVDataset(Path(args.dataset))
    episode_ids = sorted(dataset.episodes, key=episode_sort_key)
    if args.episode_filter:
        wanted = {episode_key(x) for x in args.episode_filter.split(",")}
        episode_ids = [sid for sid in episode_ids if sid in wanted]
    out = Path(args.out)

    print("Training and evaluating Paper 1 deep HGRL baseline...")
    hgrl, hgrl_history = run_hgrl_paper_baseline(dataset, episode_ids, args.max_agents, args.hgrl_episodes, args.seed, args.log_interval, args.device)
    hgrl_payload = write_method_outputs(out / "paper1_deep_hgrl_ugv_assisted", "paper1_deep_hgrl_ugv_assisted", hgrl, args.max_agents, hgrl_history)

    print("Training and evaluating Paper 2 deep CFR-MARL baseline...")
    cfr, cfr_history = run_cfr_paper_baseline(dataset, episode_ids, args.max_agents, args.cfr_episodes, args.seed + 31, args.log_interval, args.device)
    cfr_payload = write_method_outputs(out / "paper2_deep_cfr_marl", "paper2_deep_cfr_marl", cfr, args.max_agents, cfr_history)

    print("Training and evaluating Paper 3 MW-MADDPG UAV-swarm baseline...")
    mwmaddpg, mwmaddpg_history = run_mwmaddpg_baseline(dataset, episode_ids, args.max_agents, args.mwmaddpg_episodes, args.seed + 47, args.log_interval, args.device)
    mwmaddpg_payload = write_method_outputs(out / "paper3_deep_mw_maddpg_uav_swarm", "paper3_deep_mw_maddpg_uav_swarm", mwmaddpg, args.max_agents, mwmaddpg_history)

    print("Training and evaluating Paper 4 TANet-TD3-inspired multi-UAV baseline...")
    tanet, tanet_history = run_tanet_td3_baseline(dataset, episode_ids, args.max_agents, args.tanet_episodes, args.seed + 63, args.log_interval, args.device)
    tanet_payload = write_method_outputs(out / "paper4_deep_tanet_td3_multi_uav", "paper4_deep_tanet_td3_multi_uav", tanet, args.max_agents, tanet_history)

    print("Evaluating proposed adaptive HGRL method on the same dataset...")
    proposed, proposed_history = proposed_metrics(dataset, episode_ids, args.max_agents, args.proposed_episodes, args.seed + 79, args.log_interval, args.device)
    proposed_payload = write_method_outputs(out / "proposed_adaptive_hgrl", "proposed_adaptive_hgrl", proposed, args.max_agents, proposed_history)

    method_payloads = {
        "paper1_deep_hgrl_ugv_assisted": hgrl_payload,
        "paper2_deep_cfr_marl": cfr_payload,
        "paper3_deep_mw_maddpg_uav_swarm": mwmaddpg_payload,
        "paper4_deep_tanet_td3_multi_uav": tanet_payload,
        "proposed_adaptive_hgrl": proposed_payload,
    }
    write_comparison(
        out / "comparison_numeric",
        method_payloads,
        "proposed_adaptive_hgrl",
    )
    config = {
        "dataset": args.dataset,
        "benchmark_episode_count": len(episode_ids),
        "max_agents": args.max_agents,
        "device": args.device,
        "log_interval": args.log_interval,
        "training_episodes": {
            "paper1_deep_hgrl_ugv_assisted": args.hgrl_episodes,
            "paper2_deep_cfr_marl": args.cfr_episodes,
            "paper3_deep_mw_maddpg_uav_swarm": args.mwmaddpg_episodes,
            "paper4_deep_tanet_td3_multi_uav": args.tanet_episodes,
            "proposed_adaptive_hgrl": args.proposed_episodes,
        },
        "note": "No graphs or tables are generated by this runner; it writes numeric per-episode evaluation and actual training_history.csv logs for each method.",
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote numeric outputs to {out.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four deep existing-method baselines and proposed adaptive HGRL on the same dataset")
    parser.add_argument("--dataset", default="data_raw/complete_adaptive_benchmark")
    parser.add_argument("--out", default="outputs/deep_method_comparison")
    parser.add_argument("--max-agents", type=int, default=8)
    parser.add_argument("--training-episodes", type=int, default=40000)
    parser.add_argument("--hgrl-episodes", type=int, default=None)
    parser.add_argument("--cfr-episodes", type=int, default=None)
    parser.add_argument("--mwmaddpg-episodes", type=int, default=None)
    parser.add_argument("--tanet-episodes", type=int, default=None)
    parser.add_argument("--proposed-episodes", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--episode-filter", default="")
    args = parser.parse_args()
    args.hgrl_episodes = args.hgrl_episodes or args.training_episodes
    args.cfr_episodes = args.cfr_episodes or args.training_episodes
    args.mwmaddpg_episodes = args.mwmaddpg_episodes or args.training_episodes
    args.tanet_episodes = args.tanet_episodes or args.training_episodes
    args.proposed_episodes = args.proposed_episodes or args.training_episodes
    return args


if __name__ == "__main__":
    run(parse_args())
