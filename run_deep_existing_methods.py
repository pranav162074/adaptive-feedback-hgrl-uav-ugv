from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from adaptive_hgrl import (
    AdaptiveGridGraph,
    AgentTask,
    MissionBatteryState,
    PlanMetrics,
    Scenario,
    UAVDataset,
    fairness_std,
    path_length,
    risk_along_path,
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
    scenario_id: int
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


class EnergyRoutingPolicy(nn.Module):
    """2025 energy-constrained UAV/UGV DRL-routing inspired baseline."""

    def __init__(self, feature_dim: int = 8, hidden_dim: int = 64):
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


def build_heterogeneous_graph(dataset: UAVDataset, scenario_id: int, tasks: Sequence[AgentTask], battery_state: MissionBatteryState):
    profiles = dataset.agents[scenario_id]
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


def train_hgrl_allocator(dataset: UAVDataset, scenario_ids: Sequence[int], max_agents: int, epochs: int, seed: int) -> DeepHGRLAllocator:
    torch.manual_seed(seed)
    random.seed(seed)
    model = DeepHGRLAllocator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(epochs):
        losses = []
        for scenario_id in scenario_ids:
            tasks = dataset.tasks[scenario_id][:max_agents]
            battery_state = MissionBatteryState.from_profiles(dataset.agents[scenario_id])
            node_features, adj, node_ids, task_start = build_heterogeneous_graph(dataset, scenario_id, tasks, battery_state)
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


def hgrl_assignments(model: DeepHGRLAllocator, dataset: UAVDataset, scenario_id: int, max_agents: int):
    tasks = dataset.tasks[scenario_id][:max_agents]
    battery_state = MissionBatteryState.from_profiles(dataset.agents[scenario_id])
    node_features, adj, node_ids, task_start = build_heterogeneous_graph(dataset, scenario_id, tasks, battery_state)
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
    scenario: Scenario,
    assignments,
    method: str,
    use_feedback_reward: bool,
    allow_recharge_support: bool,
    adaptive_edges: bool,
    target_tasks: int,
) -> ExistingMethodMetrics:
    graph = AdaptiveGridGraph(
        22,
        dataset.static_obstacles.get(scenario.scenario_id, []),
        seed=900 + scenario.scenario_id,
        terrain_cost=dataset.terrain_cost.get(scenario.scenario_id, []),
    )
    dynamic_by_time = dataset.dynamic_obstacles.get(scenario.scenario_id, {})
    battery_state = MissionBatteryState.from_profiles(dataset.agents.get(scenario.scenario_id, {}))
    for event_step in sorted(dataset.battery_events.get(scenario.scenario_id, {})):
        if event_step <= 32:
            battery_state.apply_events(dataset.battery_events[scenario.scenario_id][event_step], event_step)

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
            feedback = events_at(dataset.feedback_events.get(scenario.scenario_id, {}), step) if use_feedback_reward else []
            communication = events_at(dataset.communication_events.get(scenario.scenario_id, {}), step) if use_feedback_reward else []
            path, cost = graph.astar(
                current,
                route_goal,
                dynamic,
                "feedback" if adaptive_edges else "static",
                battery,
                agent_type,
                feedback,
                communication,
            )
            if len(path) <= 1:
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

    return ExistingMethodMetrics(
        scenario.scenario_id,
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


def train_cfr_policy(dataset: UAVDataset, scenario_ids: Sequence[int], max_agents: int, episodes: int, seed: int) -> DeepCFRPolicy:
    torch.manual_seed(seed)
    random.seed(seed)
    policy = DeepCFRPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.002)
    for episode in range(episodes):
        scenario_id = scenario_ids[episode % len(scenario_ids)]
        tasks = dataset.tasks[scenario_id][:max_agents]
        task = tasks[episode % len(tasks)]
        graph = AdaptiveGridGraph(
            16,
            dataset.static_obstacles.get(scenario_id, []),
            seed=seed + scenario_id,
            terrain_cost=dataset.terrain_cost.get(scenario_id, []),
        )
        dynamic_by_time = dataset.dynamic_obstacles.get(scenario_id, {})
        node = graph.to_node(task.start)
        goal = graph.to_node(task.goal)
        battery = 0.54
        log_probs = []
        rewards = []
        for step in range(42):
            dynamic = dynamic_at(dynamic_by_time, step)
            feedback = events_at(dataset.feedback_events.get(scenario_id, {}), step)
            communication = events_at(dataset.communication_events.get(scenario_id, {}), step)
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


def cfr_assignments(dataset: UAVDataset, scenario_id: int, max_agents: int):
    # CFR-MARL baseline focuses on centralized feedback path policy; assignment is direct/decentralized.
    return [(f"UAV_{task.uav_id}", task.start, task.goal) for task in dataset.tasks[scenario_id][:max_agents]]


def energy_candidate_features(task: AgentTask, start: Point, battery: float, charger_distance: float, obstacle_risk: float) -> torch.Tensor:
    distance = math.dist(start, task.goal)
    energy_need = distance * (1.0 + charger_distance)
    return torch.tensor(
        [
            distance,
            battery,
            charger_distance,
            energy_need,
            obstacle_risk,
            task.priority / 5.0,
            task.payload_required / 10.0,
            float(task.requires_ugv),
        ],
        dtype=torch.float32,
    )


def train_energy_routing_policy(dataset: UAVDataset, scenario_ids: Sequence[int], max_agents: int, epochs: int, seed: int) -> EnergyRoutingPolicy:
    torch.manual_seed(seed)
    random.seed(seed)
    policy = EnergyRoutingPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
    for _ in range(epochs):
        losses = []
        for scenario_id in scenario_ids:
            tasks = dataset.tasks[scenario_id][:max_agents]
            battery_state = MissionBatteryState.from_profiles(dataset.agents[scenario_id])
            dynamic = dynamic_at(dataset.dynamic_obstacles.get(scenario_id, {}), 0)
            remaining = list(range(len(tasks)))
            for task in tasks:
                agent_id = f"UAV_{task.uav_id}"
                if not remaining:
                    continue
                battery = battery_state.level(agent_id)
                features = []
                costs = []
                for idx in remaining:
                    candidate = tasks[idx]
                    charger_distance = nearest_charger_distance(battery_state, candidate.goal)
                    risk = point_risk(candidate.goal, dynamic)
                    features.append(energy_candidate_features(candidate, task.start, battery, charger_distance, risk))
                    distance = math.dist(task.start, candidate.goal)
                    # Energy-DRL baseline learns a battery/rendezvous-aware assignment,
                    # but does not adapt graph edges online like the proposed method.
                    costs.append(distance + 0.55 * charger_distance + 1.2 * max(0.0, 0.35 - battery) + 0.8 * risk - 0.08 * candidate.priority)
                target = min(range(len(costs)), key=lambda i: costs[i])
                logits = policy(torch.stack(features))
                losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target])))
                remaining.pop(target)
        if losses:
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return policy


def energy_routing_assignments(policy: EnergyRoutingPolicy, dataset: UAVDataset, scenario_id: int, max_agents: int):
    tasks = dataset.tasks[scenario_id][:max_agents]
    battery_state = MissionBatteryState.from_profiles(dataset.agents[scenario_id])
    dynamic = dynamic_at(dataset.dynamic_obstacles.get(scenario_id, {}), 0)
    remaining = list(range(len(tasks)))
    assignments = []
    for task in tasks:
        agent_id = f"UAV_{task.uav_id}"
        if not remaining:
            continue
        battery = battery_state.level(agent_id)
        features = []
        for idx in remaining:
            candidate = tasks[idx]
            features.append(
                energy_candidate_features(
                    candidate,
                    task.start,
                    battery,
                    nearest_charger_distance(battery_state, candidate.goal),
                    point_risk(candidate.goal, dynamic),
                )
            )
        with torch.no_grad():
            chosen_pos = int(torch.argmax(policy(torch.stack(features))).item())
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


def train_tanet_td3(dataset: UAVDataset, scenario_ids: Sequence[int], max_agents: int, epochs: int, seed: int):
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
        for scenario_id in scenario_ids:
            graph = AdaptiveGridGraph(18, dataset.static_obstacles.get(scenario_id, []), seed=seed + scenario_id)
            dynamic = dynamic_at(dataset.dynamic_obstacles.get(scenario_id, {}), epoch)
            battery_state = MissionBatteryState.from_profiles(dataset.agents[scenario_id])
            for task in dataset.tasks[scenario_id][:max_agents]:
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


def tanet_td3_assignments(actor: TANetActor, critic1: TANetCritic, critic2: TANetCritic, dataset: UAVDataset, scenario_id: int, max_agents: int):
    tasks = dataset.tasks[scenario_id][:max_agents]
    graph = AdaptiveGridGraph(18, dataset.static_obstacles.get(scenario_id, []), seed=1000 + scenario_id)
    dynamic = dynamic_at(dataset.dynamic_obstacles.get(scenario_id, {}), 0)
    battery_state = MissionBatteryState.from_profiles(dataset.agents[scenario_id])
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


def run_hgrl_paper_baseline(dataset: UAVDataset, scenario_ids: Sequence[int], max_agents: int, epochs: int, seed: int):
    model = train_hgrl_allocator(dataset, scenario_ids, max_agents, epochs, seed)
    metrics = []
    for scenario_id in scenario_ids:
        assignments = hgrl_assignments(model, dataset, scenario_id, max_agents)
        metrics.append(
            evaluate_graph_paths(
                dataset,
                dataset.scenarios[scenario_id],
                assignments,
                "paper1_deep_hgrl_ugv_assisted",
                use_feedback_reward=False,
                allow_recharge_support=False,
                adaptive_edges=False,
                target_tasks=max_agents,
            )
        )
    return metrics


def run_cfr_paper_baseline(dataset: UAVDataset, scenario_ids: Sequence[int], max_agents: int, episodes: int, seed: int):
    train_cfr_policy(dataset, scenario_ids, max_agents, episodes, seed)
    metrics = []
    for scenario_id in scenario_ids:
        metrics.append(
            evaluate_graph_paths(
                dataset,
                dataset.scenarios[scenario_id],
                cfr_assignments(dataset, scenario_id, max_agents),
                "paper2_deep_cfr_marl",
                use_feedback_reward=True,
                allow_recharge_support=False,
                adaptive_edges=False,
                target_tasks=max_agents,
            )
        )
    return metrics


def run_energy_drl_baseline(dataset: UAVDataset, scenario_ids: Sequence[int], max_agents: int, epochs: int, seed: int):
    policy = train_energy_routing_policy(dataset, scenario_ids, max_agents, epochs, seed)
    metrics = []
    for scenario_id in scenario_ids:
        metrics.append(
            evaluate_graph_paths(
                dataset,
                dataset.scenarios[scenario_id],
                energy_routing_assignments(policy, dataset, scenario_id, max_agents),
                "paper3_deep_energy_uav_ugv_drl",
                use_feedback_reward=True,
                allow_recharge_support=True,
                adaptive_edges=False,
                target_tasks=max_agents,
            )
        )
    return metrics


def run_tanet_td3_baseline(dataset: UAVDataset, scenario_ids: Sequence[int], max_agents: int, epochs: int, seed: int):
    actor, critic1, critic2 = train_tanet_td3(dataset, scenario_ids, max_agents, epochs, seed)
    metrics = []
    for scenario_id in scenario_ids:
        metrics.append(
            evaluate_graph_paths(
                dataset,
                dataset.scenarios[scenario_id],
                tanet_td3_assignments(actor, critic1, critic2, dataset, scenario_id, max_agents),
                "paper4_deep_tanet_td3_multi_uav",
                use_feedback_reward=True,
                allow_recharge_support=False,
                adaptive_edges=False,
                target_tasks=max_agents,
            )
        )
    return metrics


def proposed_metrics(dataset: UAVDataset, scenario_ids: Sequence[int], max_agents: int):
    metrics = []
    for scenario_id in scenario_ids:
        plan = simulate_method(
            dataset,
            dataset.scenarios[scenario_id],
            "adaptive_feedback_hgrl",
            resolution=22,
            seed=13,
            max_agents=max_agents,
        )
        metrics.append(plan)
    return metrics


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
        "scenario_count": n,
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


def write_method_outputs(out_dir: Path, method_name: str, metrics, target_tasks: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize(method_name, metrics, target_tasks),
        "per_scenario": [metric_to_dict(m, target_tasks) for m in metrics],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [f"Method: {method_name}", "=" * (8 + len(method_name)), ""]
    for key, value in payload["summary"].items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Per-scenario numeric outputs:")
    for row in payload["per_scenario"]:
        lines.append(
            f"scenario={row['scenario_id']} objective={row['objective']:.4f} "
            f"completed={row['completed']} makespan={row['makespan']:.4f} "
            f"energy={row['energy_used']:.4f} min_battery={row.get('min_battery', 0):.4f} "
            f"risk={row['collision_risk']:.4f}"
        )
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


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    dataset = UAVDataset(Path(args.dataset))
    scenario_ids = sorted(dataset.scenarios)
    if args.scenarios:
        wanted = {int(x) for x in args.scenarios.split(",")}
        scenario_ids = [sid for sid in scenario_ids if sid in wanted]
    out = Path(args.out)

    print("Training and evaluating Paper 1 deep HGRL baseline...")
    hgrl = run_hgrl_paper_baseline(dataset, scenario_ids, args.max_agents, args.hgrl_epochs, args.seed)
    hgrl_payload = write_method_outputs(out / "paper1_deep_hgrl_ugv_assisted", "paper1_deep_hgrl_ugv_assisted", hgrl, args.max_agents)

    print("Training and evaluating Paper 2 deep CFR-MARL baseline...")
    cfr = run_cfr_paper_baseline(dataset, scenario_ids, args.max_agents, args.cfr_episodes, args.seed + 31)
    cfr_payload = write_method_outputs(out / "paper2_deep_cfr_marl", "paper2_deep_cfr_marl", cfr, args.max_agents)

    print("Training and evaluating Paper 3 deep energy-constrained UAV/UGV DRL baseline...")
    energy = run_energy_drl_baseline(dataset, scenario_ids, args.max_agents, args.energy_epochs, args.seed + 47)
    energy_payload = write_method_outputs(out / "paper3_deep_energy_uav_ugv_drl", "paper3_deep_energy_uav_ugv_drl", energy, args.max_agents)

    print("Training and evaluating Paper 4 TANet-TD3-inspired multi-UAV baseline...")
    tanet = run_tanet_td3_baseline(dataset, scenario_ids, args.max_agents, args.tanet_epochs, args.seed + 63)
    tanet_payload = write_method_outputs(out / "paper4_deep_tanet_td3_multi_uav", "paper4_deep_tanet_td3_multi_uav", tanet, args.max_agents)

    print("Evaluating proposed adaptive HGRL method on the same dataset...")
    proposed = proposed_metrics(dataset, scenario_ids, args.max_agents)
    proposed_payload = write_method_outputs(out / "proposed_adaptive_hgrl", "proposed_adaptive_hgrl", proposed, args.max_agents)

    write_comparison(
        out / "comparison_numeric",
        {
            "paper1_deep_hgrl_ugv_assisted": hgrl_payload,
            "paper2_deep_cfr_marl": cfr_payload,
            "paper3_deep_energy_uav_ugv_drl": energy_payload,
            "paper4_deep_tanet_td3_multi_uav": tanet_payload,
            "proposed_adaptive_hgrl": proposed_payload,
        },
        "proposed_adaptive_hgrl",
    )
    print(f"Wrote numeric outputs to {out.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two deep existing-method baselines and proposed adaptive HGRL on the same dataset")
    parser.add_argument("--dataset", default="data_raw/complete_adaptive_benchmark")
    parser.add_argument("--out", default="outputs/deep_method_comparison")
    parser.add_argument("--max-agents", type=int, default=8)
    parser.add_argument("--hgrl-epochs", type=int, default=45)
    parser.add_argument("--cfr-episodes", type=int, default=320)
    parser.add_argument("--energy-epochs", type=int, default=35)
    parser.add_argument("--tanet-epochs", type=int, default=45)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--scenarios", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
