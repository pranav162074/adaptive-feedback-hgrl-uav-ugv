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

from adaptive_hgrl import AdaptiveGridGraph, Point, Scenario, UAVDataset, path_length, risk_along_path


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

    def objective(self) -> float:
        return self.total_cost + 2.5 * self.makespan + 100.0 * self.collision_risk + max(0, 8 - self.completed) * 40.0


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
        alpha: float = 0.28,
        gamma: float = 0.92,
        epsilon: float = 0.32,
        seed: int = 11,
    ):
        self.graph = graph
        self.dynamic_by_time = dynamic_by_time
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.random = random.Random(seed)
        self.q: Dict[Tuple[Node, Node, int, int, int], float] = defaultdict(float)
        self.graph_updates = 0

    def state_key(self, node: Node, goal: Node, time_step: int, battery: float) -> Tuple[Node, Node, int, int]:
        dynamic = self.dynamic_by_time.get(time_step % max(1, len(self.dynamic_by_time)), [])
        risk = self.graph.obstacle_risk(self.graph.to_point(node), dynamic, 0.08)
        risk_bucket = min(3, int(risk * 4))
        battery_bucket = min(4, max(0, int(battery * 5)))
        return node, goal, risk_bucket, battery_bucket

    def valid_actions(self, node: Node, time_step: int, battery: float) -> List[int]:
        dynamic = self.dynamic_by_time.get(time_step % max(1, len(self.dynamic_by_time)), [])
        valid: List[int] = []
        for idx, action in enumerate(ACTIONS):
            nxt = (node[0] + action[0], node[1] + action[1])
            if 0 <= nxt[0] < self.graph.resolution and 0 <= nxt[1] < self.graph.resolution:
                if self.graph.edge_cost(node, nxt, dynamic, "feedback", battery) is not None:
                    valid.append(idx)
        return valid

    def choose_action(self, state: Tuple[Node, Node, int, int], valid: Sequence[int], training: bool) -> int:
        if not valid:
            return 0
        if training and self.random.random() < self.epsilon:
            return self.random.choice(list(valid))
        return max(valid, key=lambda a: self.q[(state[0], state[1], state[2], state[3], a)])

    def step_reward(self, node: Node, nxt: Node, goal: Node, time_step: int, battery: float) -> Tuple[float, float]:
        dynamic = self.dynamic_by_time.get(time_step % max(1, len(self.dynamic_by_time)), [])
        edge_cost = self.graph.edge_cost(node, nxt, dynamic, "feedback", battery)
        if edge_cost is None:
            return -50.0, 1.0
        point = self.graph.to_point(nxt)
        risk = self.graph.obstacle_risk(point, dynamic, 0.07)
        progress = math.dist(self.graph.to_point(node), self.graph.to_point(goal)) - math.dist(point, self.graph.to_point(goal))
        terminal = 35.0 if nxt == goal else 0.0
        centralized_feedback_penalty = 18.0 * risk
        reward = 8.0 * progress + terminal - edge_cost - centralized_feedback_penalty - 0.04 * (1.0 - battery)
        return reward, risk

    def train_episode(self, start: Point, goal: Point, max_steps: int = 120) -> float:
        node = self.graph.to_node(start)
        goal_node = self.graph.to_node(goal)
        battery = 1.0
        total_reward = 0.0
        traversed: List[Point] = [self.graph.to_point(node)]
        for time_step in range(max_steps):
            state = self.state_key(node, goal_node, time_step, battery)
            valid = self.valid_actions(node, time_step, battery)
            if not valid:
                total_reward -= 40.0
                break
            action_idx = self.choose_action(state, valid, training=True)
            action = ACTIONS[action_idx]
            nxt = (node[0] + action[0], node[1] + action[1])
            reward, risk = self.step_reward(node, nxt, goal_node, time_step, battery)
            next_state = self.state_key(nxt, goal_node, time_step + 1, battery)
            next_valid = self.valid_actions(nxt, time_step + 1, battery)
            best_next = max((self.q[(next_state[0], next_state[1], next_state[2], next_state[3], a)] for a in next_valid), default=0.0)
            old = self.q[(state[0], state[1], state[2], state[3], action_idx)]
            self.q[(state[0], state[1], state[2], state[3], action_idx)] = old + self.alpha * (reward + self.gamma * best_next - old)
            node = nxt
            traversed.append(self.graph.to_point(node))
            battery = max(0.0, battery - 0.006)
            total_reward += reward
            if risk > 0.45:
                dynamic = self.dynamic_by_time.get(time_step % max(1, len(self.dynamic_by_time)), [])
                self.graph_updates += self.graph.update_feedback(traversed[-2:], dynamic)
            if node == goal_node:
                break
        self.epsilon = max(0.04, self.epsilon * 0.996)
        return total_reward

    def rollout(self, start: Point, goal: Point, max_steps: int = 120) -> Tuple[List[Point], float, bool]:
        node = self.graph.to_node(start)
        goal_node = self.graph.to_node(goal)
        battery = 1.0
        path = [self.graph.to_point(node)]
        reward_total = 0.0
        for time_step in range(max_steps):
            state = self.state_key(node, goal_node, time_step, battery)
            valid = self.valid_actions(node, time_step, battery)
            if not valid:
                break
            action_idx = self.choose_action(state, valid, training=False)
            action = ACTIONS[action_idx]
            nxt = (node[0] + action[0], node[1] + action[1])
            reward, risk = self.step_reward(node, nxt, goal_node, time_step, battery)
            reward_total += reward
            node = nxt
            path.append(self.graph.to_point(node))
            battery = max(0.0, battery - 0.006)
            if risk > 0.45:
                dynamic = self.dynamic_by_time.get(time_step % max(1, len(self.dynamic_by_time)), [])
                self.graph_updates += self.graph.update_feedback(path[-2:], dynamic)
            if node == goal_node:
                return path, reward_total, True
        return path, reward_total, node == goal_node


def assign_tasks_by_learned_value(tasks, learner: CentralizedFeedbackGraphRL):
    remaining = [task.goal for task in tasks]
    assigned = []
    for task in tasks:
        best_i = 0
        best_value = -float("inf")
        start_node = learner.graph.to_node(task.start)
        for i, goal in enumerate(remaining):
            goal_node = learner.graph.to_node(goal)
            state = learner.state_key(start_node, goal_node, 0, 1.0)
            valid = learner.valid_actions(start_node, 0, 1.0)
            value = max((learner.q[(state[0], state[1], state[2], state[3], a)] for a in valid), default=0.0)
            value -= 0.25 * math.dist(task.start, goal)
            if value > best_value:
                best_value = value
                best_i = i
        assigned.append((task.uav_id, task.start, remaining.pop(best_i)))
    return assigned


def train_and_evaluate(dataset: UAVDataset, scenario: Scenario, resolution: int, episodes: int, max_agents: int, seed: int) -> RLRunMetrics:
    graph = AdaptiveGridGraph(resolution, dataset.static_obstacles.get(scenario.scenario_id, []), seed=seed + scenario.scenario_id)
    dynamic_by_time = dataset.dynamic_obstacles.get(scenario.scenario_id, {})
    if not dynamic_by_time:
        dynamic_by_time = {0: []}
    learner = CentralizedFeedbackGraphRL(graph, dynamic_by_time, seed=seed + scenario.scenario_id)
    tasks = dataset.tasks[scenario.scenario_id][:max_agents]

    rewards: List[float] = []
    for ep in range(episodes):
        task = tasks[ep % len(tasks)]
        rewards.append(learner.train_episode(task.start, task.goal))

    lengths: List[float] = []
    risks: List[float] = []
    costs: List[float] = []
    completed = 0
    assigned = assign_tasks_by_learned_value(tasks, learner)
    all_dynamic = [point for points in dynamic_by_time.values() for point in points[:20]]
    for _, start, goal in assigned:
        path, reward, done = learner.rollout(start, goal)
        length = path_length(path)
        lengths.append(length)
        risks.append(risk_along_path(path, all_dynamic))
        costs.append(max(0.0, -reward / 10.0) + length)
        completed += int(done)

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
