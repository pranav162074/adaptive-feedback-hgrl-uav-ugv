from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from adaptive_hgrl import AdaptiveGridGraph, MissionBatteryState, Point, Scenario, UAVDataset, path_length, risk_along_path, terrain_factor


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
class RLRunMetrics:
    scenario_id: int
    method: str
    reward_mean: float
    total_cost: float
    makespan: float
    avg_path_length: float
    collision_risk: float
    completed: int
    graph_updates: int
    episodes: int
    min_battery: float = 0.0
    battery_warnings: int = 0
    recharge_visits: int = 0

    def objective(self) -> float:
        return (
            self.total_cost
            + 2.5 * self.makespan
            + 100.0 * self.collision_risk
            + max(0, 8 - self.completed) * 40.0
            + max(0.0, 0.30 - self.min_battery) * 75.0
            + self.battery_warnings * 2.5
        )


class CentralizedFeedbackGraphRL:
    """Centralized graph reinforcement learner with adaptive feedback rewards.

    This is intentionally lightweight: it uses tabular Q-learning instead of a
    deep GNN so it runs on a normal internship laptop without PyTorch. The graph
    state, centralized reward, temporal-difference update, and feedback-driven
    topology changes are all explicit.
    """

    def __init__(
        self,
        graph: AdaptiveGridGraph,
        dynamic_by_time: Dict[int, List[Point]],
        feedback_by_time=None,
        communication_by_time=None,
        alpha: float = 0.28,
        gamma: float = 0.92,
        epsilon: float = 0.32,
        seed: int = 11,
    ):
        self.graph = graph
        self.dynamic_by_time = dynamic_by_time
        self.feedback_by_time = feedback_by_time or {}
        self.communication_by_time = communication_by_time or {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.random = random.Random(seed)
        self.q: Dict[Tuple[Node, Node, int, int, int, int], float] = defaultdict(float)
        self.graph_updates = 0

    def time_key(self, time_step: int) -> int:
        keys = sorted(self.dynamic_by_time) or [0]
        return keys[time_step % len(keys)]

    def state_key(self, node: Node, goal: Node, time_step: int, battery: float) -> Tuple[Node, Node, int, int, int]:
        dynamic = self.dynamic_by_time.get(self.time_key(time_step), [])
        risk = self.graph.obstacle_risk(self.graph.to_point(node), dynamic, 0.08)
        risk_bucket = min(3, int(risk * 4))
        battery_bucket = min(4, max(0, int(battery * 5)))
        support_bucket = min(3, int(math.dist(self.graph.to_point(node), self.graph.to_point(goal)) * 4))
        return node, goal, risk_bucket, battery_bucket, support_bucket

    def valid_actions(self, node: Node, time_step: int, battery: float, agent_type: str = "UAV") -> List[int]:
        key = self.time_key(time_step)
        dynamic = self.dynamic_by_time.get(key, [])
        feedback = self.feedback_by_time.get(key, [])
        communication = self.communication_by_time.get(key, [])
        valid: List[int] = []
        for idx, action in enumerate(ACTIONS):
            nxt = (node[0] + action[0], node[1] + action[1])
            if 0 <= nxt[0] < self.graph.resolution and 0 <= nxt[1] < self.graph.resolution:
                if self.graph.edge_cost(node, nxt, dynamic, "feedback", battery, agent_type, feedback, communication) is not None:
                    valid.append(idx)
        return valid

    def choose_action(self, state: Tuple[Node, Node, int, int, int], valid: Sequence[int], training: bool) -> int:
        if not valid:
            return 0
        if training and self.random.random() < self.epsilon:
            return self.random.choice(list(valid))
        return max(valid, key=lambda a: self.q[(state[0], state[1], state[2], state[3], state[4], a)])

    def step_reward(self, node: Node, nxt: Node, goal: Node, time_step: int, battery: float, agent_type: str = "UAV") -> Tuple[float, float]:
        key = self.time_key(time_step)
        dynamic = self.dynamic_by_time.get(key, [])
        feedback = self.feedback_by_time.get(key, [])
        communication = self.communication_by_time.get(key, [])
        edge_cost = self.graph.edge_cost(node, nxt, dynamic, "feedback", battery, agent_type, feedback, communication)
        if edge_cost is None:
            return -50.0, 1.0
        point = self.graph.to_point(nxt)
        risk = max(
            self.graph.obstacle_risk(point, dynamic, 0.07),
            self.graph.spatial_event_risk(point, feedback),
            0.65 * self.graph.spatial_event_risk(point, communication),
        )
        progress = math.dist(self.graph.to_point(node), self.graph.to_point(goal)) - math.dist(point, self.graph.to_point(goal))
        terminal = 35.0 if nxt == goal else 0.0
        centralized_feedback_penalty = 18.0 * risk
        low_battery_penalty = 3.8 * max(0.0, 0.30 - battery)
        reward = 8.0 * progress + terminal - edge_cost - centralized_feedback_penalty - low_battery_penalty
        return reward, risk

    def train_episode(self, agent_id: str, start: Point, goal: Point, battery_state: MissionBatteryState, max_steps: int = 120) -> float:
        node = self.graph.to_node(start)
        goal_node = self.graph.to_node(goal)
        profile = battery_state.profiles.get(agent_id)
        agent_type = profile.type if profile else "UAV"
        battery = battery_state.level(agent_id)
        total_reward = 0.0
        traversed: List[Point] = [self.graph.to_point(node)]
        for time_step in range(max_steps):
            state = self.state_key(node, goal_node, time_step, battery)
            valid = self.valid_actions(node, time_step, battery, agent_type)
            if not valid:
                total_reward -= 40.0
                break
            action_idx = self.choose_action(state, valid, training=True)
            action = ACTIONS[action_idx]
            nxt = (node[0] + action[0], node[1] + action[1])
            reward, risk = self.step_reward(node, nxt, goal_node, time_step, battery, agent_type)
            next_state = self.state_key(nxt, goal_node, time_step + 1, battery)
            next_valid = self.valid_actions(nxt, time_step + 1, battery, agent_type)
            best_next = max((self.q[(next_state[0], next_state[1], next_state[2], next_state[3], next_state[4], a)] for a in next_valid), default=0.0)
            old = self.q[(state[0], state[1], state[2], state[3], state[4], action_idx)]
            self.q[(state[0], state[1], state[2], state[3], state[4], action_idx)] = old + self.alpha * (reward + self.gamma * best_next - old)
            node = nxt
            traversed.append(self.graph.to_point(node))
            executed = traversed[-2:]
            battery_state.drain(agent_id, path_length(executed), terrain_factor(self.graph, executed), time_step)
            battery_state.maybe_recharge(agent_id, self.graph.to_point(node), time_step)
            battery = battery_state.level(agent_id)
            total_reward += reward
            if risk > 0.45 or battery < 0.25:
                key = self.time_key(time_step)
                dynamic = self.dynamic_by_time.get(key, [])
                self.graph_updates += self.graph.update_feedback(traversed[-2:], dynamic, battery, self.feedback_by_time.get(key, []), self.communication_by_time.get(key, []))
            if node == goal_node:
                break
        self.epsilon = max(0.04, self.epsilon * 0.996)
        return total_reward

    def rollout(self, agent_id: str, start: Point, goal: Point, battery_state: MissionBatteryState, max_steps: int = 120) -> Tuple[List[Point], float, bool]:
        node = self.graph.to_node(start)
        goal_node = self.graph.to_node(goal)
        profile = battery_state.profiles.get(agent_id)
        agent_type = profile.type if profile else "UAV"
        battery = battery_state.level(agent_id)
        path = [self.graph.to_point(node)]
        reward_total = 0.0
        for time_step in range(max_steps):
            state = self.state_key(node, goal_node, time_step, battery)
            valid = self.valid_actions(node, time_step, battery, agent_type)
            if not valid:
                break
            action_idx = self.choose_action(state, valid, training=False)
            action = ACTIONS[action_idx]
            nxt = (node[0] + action[0], node[1] + action[1])
            reward, risk = self.step_reward(node, nxt, goal_node, time_step, battery, agent_type)
            reward_total += reward
            node = nxt
            path.append(self.graph.to_point(node))
            executed = path[-2:]
            battery_state.drain(agent_id, path_length(executed), terrain_factor(self.graph, executed), time_step)
            battery_state.maybe_recharge(agent_id, self.graph.to_point(node), time_step)
            battery = battery_state.level(agent_id)
            if risk > 0.45 or battery < 0.25:
                key = self.time_key(time_step)
                dynamic = self.dynamic_by_time.get(key, [])
                self.graph_updates += self.graph.update_feedback(path[-2:], dynamic, battery, self.feedback_by_time.get(key, []), self.communication_by_time.get(key, []))
            if node == goal_node:
                return path, reward_total, True
        return path, reward_total, node == goal_node


def assign_tasks_by_learned_value(tasks, learner: CentralizedFeedbackGraphRL, battery_state: MissionBatteryState):
    remaining = [task.goal for task in tasks]
    assigned = []
    for task in tasks:
        best_i = 0
        best_value = -float("inf")
        agent_id = f"UAV_{task.uav_id}"
        battery = battery_state.level(agent_id)
        start_node = learner.graph.to_node(task.start)
        for i, goal in enumerate(remaining):
            goal_node = learner.graph.to_node(goal)
            state = learner.state_key(start_node, goal_node, 0, battery)
            valid = learner.valid_actions(start_node, 0, battery)
            value = max((learner.q[(state[0], state[1], state[2], state[3], state[4], a)] for a in valid), default=0.0)
            value -= 0.25 * math.dist(task.start, goal)
            value -= max(0.0, 0.42 - battery) * math.dist(task.start, goal)
            value += 0.08 * task.priority
            if value > best_value:
                best_value = value
                best_i = i
        assigned.append((agent_id, task.start, remaining.pop(best_i)))
    return assigned


def train_and_evaluate(dataset: UAVDataset, scenario: Scenario, resolution: int, episodes: int, max_agents: int, seed: int) -> RLRunMetrics:
    graph = AdaptiveGridGraph(
        resolution,
        dataset.static_obstacles.get(scenario.scenario_id, []),
        seed=seed + scenario.scenario_id,
        terrain_cost=dataset.terrain_cost.get(scenario.scenario_id, []),
    )
    dynamic_by_time = dataset.dynamic_obstacles.get(scenario.scenario_id, {})
    if not dynamic_by_time:
        dynamic_by_time = {0: []}
    learner = CentralizedFeedbackGraphRL(
        graph,
        dynamic_by_time,
        dataset.feedback_events.get(scenario.scenario_id, {}),
        dataset.communication_events.get(scenario.scenario_id, {}),
        seed=seed + scenario.scenario_id,
    )
    tasks = dataset.tasks[scenario.scenario_id][:max_agents]
    train_battery_state = MissionBatteryState.from_profiles(dataset.agents.get(scenario.scenario_id, {}))

    rewards: List[float] = []
    for ep in range(episodes):
        task = tasks[ep % len(tasks)]
        step = ep % 80
        train_battery_state.apply_events(dataset.battery_events.get(scenario.scenario_id, {}).get(step, []), step)
        rewards.append(learner.train_episode(f"UAV_{task.uav_id}", task.start, task.goal, train_battery_state))

    lengths: List[float] = []
    risks: List[float] = []
    costs: List[float] = []
    completed = 0
    eval_battery_state = MissionBatteryState.from_profiles(dataset.agents.get(scenario.scenario_id, {}))
    for event_step in sorted(dataset.battery_events.get(scenario.scenario_id, {})):
        if event_step <= 32:
            eval_battery_state.apply_events(dataset.battery_events[scenario.scenario_id][event_step], event_step)
    assigned = assign_tasks_by_learned_value(tasks, learner, eval_battery_state)
    all_dynamic = [point for points in dynamic_by_time.values() for point in points[:20]]
    for agent_id, start, goal in assigned:
        path, reward, done = learner.rollout(agent_id, start, goal, eval_battery_state)
        length = path_length(path)
        lengths.append(length)
        risks.append(risk_along_path(path, all_dynamic))
        costs.append(max(0.0, -reward / 10.0) + length)
        completed += int(done and eval_battery_state.level(agent_id) > 0.05)

    return RLRunMetrics(
        scenario.scenario_id,
        "centralized_feedback_graph_rl",
        sum(rewards[-max_agents:]) / max(1, min(max_agents, len(rewards))),
        sum(costs),
        max(lengths) if lengths else 0.0,
        sum(lengths) / max(1, len(lengths)),
        sum(risks) / max(1, len(risks)),
        completed,
        learner.graph_updates,
        episodes,
        eval_battery_state.min_level(),
        eval_battery_state.warnings,
        eval_battery_state.recharge_visits,
    )


def write_results(out: Path, rows: Sequence[RLRunMetrics]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rl_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].__dict__) + ["objective"])
        writer.writeheader()
        for row in rows:
            data = row.__dict__.copy()
            data["objective"] = row.objective()
            writer.writerow(data)
    summary = {
        "method": "centralized_feedback_graph_rl",
        "objective_mean": sum(r.objective() for r in rows) / len(rows),
        "collision_risk_mean": sum(r.collision_risk for r in rows) / len(rows),
        "completed_mean": sum(r.completed for r in rows) / len(rows),
        "reward_mean": sum(r.reward_mean for r in rows) / len(rows),
        "graph_updates_total": sum(r.graph_updates for r in rows),
        "min_battery_mean": sum(r.min_battery for r in rows) / len(rows),
        "battery_warnings_total": sum(r.battery_warnings for r in rows),
        "recharge_visits_total": sum(r.recharge_visits for r in rows),
        "episodes_per_scenario": rows[0].episodes,
    }
    (out / "rl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    dataset = UAVDataset(Path(args.dataset))
    scenario_ids = sorted(dataset.scenarios)
    if args.scenarios:
        wanted = {int(s) for s in args.scenarios.split(",")}
        scenario_ids = [sid for sid in scenario_ids if sid in wanted]
    rows = [
        train_and_evaluate(dataset, dataset.scenarios[sid], args.resolution, args.episodes, args.max_agents, args.seed)
        for sid in scenario_ids
    ]
    out = Path(args.out)
    write_results(out, rows)
    print(f"Wrote RL results to {out.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Centralized feedback-driven graph reinforcement learning trainer")
    parser.add_argument("--dataset", default="data_raw/uav_dataset/uav_dataset")
    parser.add_argument("--out", default="outputs/centralized_graph_rl")
    parser.add_argument("--resolution", type=int, default=18)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--max-agents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--scenarios", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
