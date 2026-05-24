from __future__ import annotations

import argparse
import csv
import heapq
import html
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float]
Node = Tuple[int, int]
Edge = Tuple[Node, Node]


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    density: float
    clustered: str
    dynamic: str


@dataclass(frozen=True)
class AgentTask:
    uav_id: int
    start: Point
    goal: Point


@dataclass
class PlanMetrics:
    scenario_id: int
    method: str
    total_cost: float
    makespan: float
    avg_path_length: float
    collision_risk: float
    energy_used: float
    completed: int
    replans: int
    graph_updates: int
    fairness_std: float

    def objective(self) -> float:
        incomplete = max(0, 10 - self.completed) * 25.0
        return (
            self.total_cost
            + 2.5 * self.makespan
            + 80.0 * self.collision_risk
            + 0.8 * self.energy_used
            + 5.0 * self.fairness_std
            + incomplete
        )


class UAVDataset:
    def __init__(self, root: Path):
        self.root = root
        self.scenarios = self._load_scenarios()
        self.tasks = self._load_tasks()
        self.static_obstacles = self._load_static_obstacles()
        self.dynamic_obstacles = self._load_dynamic_obstacles()

    def _load_scenarios(self) -> Dict[int, Scenario]:
        rows: Dict[int, Scenario] = {}
        with (self.root / "scenarios.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                sid = int(row["scenario_id"])
                rows[sid] = Scenario(sid, float(row["density"]), row["clustered"], row["dynamic"])
        return rows

    def _load_tasks(self) -> Dict[int, List[AgentTask]]:
        tasks: Dict[int, List[AgentTask]] = defaultdict(list)
        with (self.root / "uav_positions.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                sid = int(row["scenario_id"])
                tasks[sid].append(
                    AgentTask(
                        int(row["uav_id"]),
                        (float(row["start_x"]), float(row["start_y"])),
                        (float(row["goal_x"]), float(row["goal_y"])),
                    )
                )
        return tasks

    def _load_static_obstacles(self) -> Dict[int, List[Point]]:
        obs: Dict[int, List[Point]] = defaultdict(list)
        with (self.root / "static_obstacles.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                obs[int(row["scenario_id"])].append((float(row["x"]), float(row["y"])))
        return obs

    def _load_dynamic_obstacles(self) -> Dict[int, Dict[int, List[Point]]]:
        obs: Dict[int, Dict[int, List[Point]]] = defaultdict(lambda: defaultdict(list))
        with (self.root / "dynamic_obstacles.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                sid = int(row["scenario_id"])
                time_step = int(row["time_step"])
                obs[sid][time_step].append((float(row["x"]), float(row["y"])))
        return obs


class AdaptiveGridGraph:
    def __init__(self, resolution: int, static_obstacles: Sequence[Point], seed: int = 7):
        self.resolution = resolution
        self.static_obstacles = list(static_obstacles)
        self.feedback_penalty: Dict[Edge, float] = defaultdict(float)
        self.blocked_feedback: Dict[Edge, int] = defaultdict(int)
        self.random = random.Random(seed)
        self.nodes: List[Node] = [(x, y) for x in range(resolution) for y in range(resolution)]
        self.neighbor_delta = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        self.static_risk = self._precompute_static_risk()

    def to_node(self, p: Point) -> Node:
        n = self.resolution - 1
        return (min(n, max(0, round(p[0] * n))), min(n, max(0, round(p[1] * n))))

    def to_point(self, n: Node) -> Point:
        d = self.resolution - 1
        return (n[0] / d, n[1] / d)

    def neighbors(self, node: Node) -> Iterable[Node]:
        for dx, dy in self.neighbor_delta:
            nxt = (node[0] + dx, node[1] + dy)
            if 0 <= nxt[0] < self.resolution and 0 <= nxt[1] < self.resolution:
                yield nxt

    def edge_key(self, a: Node, b: Node) -> Edge:
        return (a, b) if a <= b else (b, a)

    def obstacle_risk(self, point: Point, obstacles: Sequence[Point], radius: float) -> float:
        if not obstacles:
            return 0.0
        best = min(math.dist(point, obstacle) for obstacle in obstacles)
        if best >= radius:
            return 0.0
        return (radius - best) / radius

    def _precompute_static_risk(self) -> Dict[Node, float]:
        radius = 0.035
        cell_radius = max(1, math.ceil(radius * (self.resolution - 1)))
        risk: Dict[Node, float] = defaultdict(float)
        for obstacle in self.static_obstacles:
            ox, oy = self.to_node(obstacle)
            for dx in range(-cell_radius, cell_radius + 1):
                for dy in range(-cell_radius, cell_radius + 1):
                    node = (ox + dx, oy + dy)
                    if 0 <= node[0] < self.resolution and 0 <= node[1] < self.resolution:
                        d = math.dist(self.to_point(node), obstacle)
                        if d < radius:
                            risk[node] = max(risk[node], (radius - d) / radius)
        return risk

    def edge_cost(
        self,
        a: Node,
        b: Node,
        dynamic_obstacles: Sequence[Point],
        mode: str,
        battery: float,
    ) -> Optional[float]:
        pa = self.to_point(a)
        pb = self.to_point(b)
        mid = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
        static_risk = max(self.static_risk.get(a, 0.0), self.static_risk.get(b, 0.0))
        dynamic_risk = self.obstacle_risk(mid, dynamic_obstacles, 0.055)
        if static_risk > 0.92:
            return None
        if mode == "static":
            dynamic_risk = 0.0
            feedback = 0.0
            blocked = 0
        else:
            feedback = self.feedback_penalty[self.edge_key(a, b)]
            blocked = self.blocked_feedback[self.edge_key(a, b)]
        if blocked > 1:
            return None
        base = math.dist(pa, pb)
        battery_pressure = max(0.0, 0.35 - battery) * 0.6
        return base * (1.0 + 9.0 * static_risk + 12.0 * dynamic_risk + feedback + battery_pressure)

    def astar(
        self,
        start: Point,
        goal: Point,
        dynamic_obstacles: Sequence[Point],
        mode: str,
        battery: float,
    ) -> Tuple[List[Point], float]:
        src = self.to_node(start)
        dst = self.to_node(goal)
        frontier: List[Tuple[float, Node]] = [(0.0, src)]
        came_from: Dict[Node, Optional[Node]] = {src: None}
        cost_so_far: Dict[Node, float] = {src: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == dst:
                break
            for nxt in self.neighbors(current):
                step = self.edge_cost(current, nxt, dynamic_obstacles, mode, battery)
                if step is None:
                    continue
                new_cost = cost_so_far[current] + step
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + math.dist(self.to_point(nxt), self.to_point(dst))
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current
        if dst not in came_from:
            return [start], 1_000.0
        nodes: List[Node] = []
        cur: Optional[Node] = dst
        while cur is not None:
            nodes.append(cur)
            cur = came_from[cur]
        nodes.reverse()
        return [self.to_point(n) for n in nodes], cost_so_far[dst]

    def update_feedback(self, path: Sequence[Point], dynamic_obstacles: Sequence[Point]) -> int:
        updates = 0
        for a_pt, b_pt in zip(path, path[1:]):
            a = self.to_node(a_pt)
            b = self.to_node(b_pt)
            mid = ((a_pt[0] + b_pt[0]) / 2.0, (a_pt[1] + b_pt[1]) / 2.0)
            risk = self.obstacle_risk(mid, dynamic_obstacles, 0.07)
            edge = self.edge_key(a, b)
            if risk > 0.45:
                self.feedback_penalty[edge] += 0.8 * risk
                self.blocked_feedback[edge] += 1
                updates += 1
            elif self.feedback_penalty[edge] > 0:
                self.feedback_penalty[edge] *= 0.92
        return updates


def path_length(path: Sequence[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def risk_along_path(path: Sequence[Point], obstacles: Sequence[Point]) -> float:
    if not path or not obstacles:
        return 0.0
    total = 0.0
    for point in path:
        d = min(math.dist(point, o) for o in obstacles)
        total += max(0.0, 0.055 - d) / 0.055
    return total / max(1, len(path))


def segment_execution_cost(graph: AdaptiveGridGraph, path: Sequence[Point], dynamic_obstacles: Sequence[Point], mode: str, battery: float) -> float:
    if len(path) < 2:
        return 0.0
    cost = 0.0
    for a_pt, b_pt in zip(path, path[1:]):
        edge_cost = graph.edge_cost(graph.to_node(a_pt), graph.to_node(b_pt), dynamic_obstacles, mode, battery)
        cost += edge_cost if edge_cost is not None else 100.0
    return cost


def fairness_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def assign_tasks_hierarchical(tasks: List[AgentTask], graph: AdaptiveGridGraph, dynamic: Sequence[Point], mode: str) -> List[AgentTask]:
    remaining_goals = [task.goal for task in tasks]
    assigned: List[AgentTask] = []
    for task in sorted(tasks, key=lambda t: t.uav_id):
        best_idx = 0
        best_cost = float("inf")
        for idx, goal in enumerate(remaining_goals):
            _, cost = graph.astar(task.start, goal, dynamic, mode, battery=1.0)
            heterogeneity_bonus = 0.94 if task.uav_id % 3 == 0 else 1.0
            score = cost * heterogeneity_bonus
            if score < best_cost:
                best_cost = score
                best_idx = idx
        assigned.append(AgentTask(task.uav_id, task.start, remaining_goals.pop(best_idx)))
    return assigned


def simulate_method(
    dataset: UAVDataset,
    scenario: Scenario,
    method: str,
    resolution: int,
    seed: int,
    max_agents: int,
) -> PlanMetrics:
    static = dataset.static_obstacles.get(scenario.scenario_id, [])
    dynamic_by_time = dataset.dynamic_obstacles.get(scenario.scenario_id, {})
    graph = AdaptiveGridGraph(resolution, static, seed + scenario.scenario_id)
    tasks = dataset.tasks[scenario.scenario_id][:max_agents]
    initial_dynamic = dynamic_by_time.get(0, [])

    if method == "static_graph":
        assigned = tasks
        mode = "static"
        replanning_steps = [0]
    elif method == "central_feedback_static_topology":
        assigned = assign_tasks_hierarchical(tasks, graph, initial_dynamic, "static")
        mode = "static"
        replanning_steps = [0, 20]
    else:
        assigned = assign_tasks_hierarchical(tasks, graph, initial_dynamic, "feedback")
        mode = "feedback"
        replanning_steps = [0, 8, 16, 24, 32]

    path_costs: List[float] = []
    lengths: List[float] = []
    risks: List[float] = []
    energy: List[float] = []
    completed = 0
    graph_updates = 0
    replans = 0

    for task in assigned:
        current = task.start
        battery = 1.0 - 0.04 * (task.uav_id % 4)
        full_path: List[Point] = [current]
        accumulated_cost = 0.0
        for step in replanning_steps:
            dynamic = dynamic_by_time.get(step, [])
            segment, cost = graph.astar(current, task.goal, dynamic, mode, battery)
            replans += 1 if step else 0
            if len(segment) > 1:
                if method == "adaptive_feedback_hgrl":
                    lookahead = max(2, min(len(segment), 7))
                    partial = segment[1:lookahead]
                    current = partial[-1]
                    full_path.extend(partial)
                    accumulated_cost += segment_execution_cost(graph, [full_path[-len(partial)-1]] + partial, dynamic, mode, battery)
                    graph_updates += graph.update_feedback(partial, dynamic)
                    battery -= path_length(partial) * 0.35
                else:
                    accumulated_cost += cost
                    full_path.extend(segment[1:])
                    current = task.goal
                    if method == "central_feedback_static_topology":
                        graph_updates += graph.update_feedback(segment, dynamic)
                    break
        if current != task.goal:
            dynamic = dynamic_by_time.get(max(replanning_steps), [])
            tail, cost = graph.astar(current, task.goal, dynamic, mode, battery)
            accumulated_cost += cost
            full_path.extend(tail[1:])
        length = path_length(full_path)
        all_dynamic = [p for t in replanning_steps for p in dynamic_by_time.get(t, [])]
        risk = risk_along_path(full_path, all_dynamic)
        battery_used = min(1.0, length * (0.42 if task.uav_id % 3 else 0.34))
        if accumulated_cost < 999.0 and risk < 0.75:
            completed += 1
        path_costs.append(accumulated_cost)
        lengths.append(length)
        risks.append(risk)
        energy.append(battery_used)

    makespan = max(lengths) if lengths else 0.0
    return PlanMetrics(
        scenario.scenario_id,
        method,
        sum(path_costs),
        makespan,
        sum(lengths) / max(1, len(lengths)),
        sum(risks) / max(1, len(risks)),
        sum(energy),
        completed,
        replans,
        graph_updates,
        fairness_std(lengths),
    )


def write_csv(path: Path, rows: Sequence[PlanMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario_id",
        "method",
        "total_cost",
        "makespan",
        "avg_path_length",
        "collision_risk",
        "energy_used",
        "completed",
        "replans",
        "graph_updates",
        "fairness_std",
        "objective",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = row.__dict__.copy()
            data["objective"] = row.objective()
            writer.writerow(data)


def svg_bar_chart(path: Path, title: str, rows: Sequence[PlanMetrics], metric: str) -> None:
    by_method: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = row.objective() if metric == "objective" else float(getattr(row, metric))
        by_method[row.method].append(value)
    labels = list(by_method)
    values = [sum(v) / len(v) for v in by_method.values()]
    width, height = 900, 520
    margin = 70
    max_value = max(values) if values else 1.0
    colors = {
        "static_graph": "#52616b",
        "central_feedback_static_topology": "#c7792f",
        "adaptive_feedback_hgrl": "#138a72",
    }
    bars = []
    bar_w = (width - 2 * margin) / max(1, len(values)) * 0.62
    for i, (label, value) in enumerate(zip(labels, values)):
        x = margin + i * ((width - 2 * margin) / len(values)) + bar_w * 0.3
        h = (height - 2 * margin) * value / max_value
        y = height - margin - h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="5" fill="{colors.get(label, "#777")}"/>'
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 35}" text-anchor="middle" font-size="14">{html.escape(label)}</text>'
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="14">{value:.3f}</text>'
        )
    content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f8faf8"/>
<text x="{width / 2}" y="34" text-anchor="middle" font-size="24" font-family="Arial" fill="#172026">{html.escape(title)}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#172026"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#172026"/>
<text x="22" y="{height / 2}" transform="rotate(-90 22 {height / 2})" text-anchor="middle" font-size="14" font-family="Arial">{html.escape(metric)}</text>
<g font-family="Arial" fill="#172026">{''.join(bars)}</g>
</svg>"""
    path.write_text(content, encoding="utf-8")


def svg_scenario_map(path: Path, dataset: UAVDataset, scenario_id: int, resolution: int) -> None:
    scenario = dataset.scenarios[scenario_id]
    graph = AdaptiveGridGraph(resolution, dataset.static_obstacles[scenario_id], seed=3)
    tasks = dataset.tasks[scenario_id][:8]
    dynamic = dataset.dynamic_obstacles.get(scenario_id, {}).get(8, [])
    width = height = 720

    def xy(p: Point) -> Tuple[float, float]:
        return (p[0] * (width - 80) + 40, height - (p[1] * (height - 80) + 40))

    paths = []
    for task in assign_tasks_hierarchical(tasks, graph, dynamic, "feedback"):
        path_points, _ = graph.astar(task.start, task.goal, dynamic, "feedback", 1.0)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, path_points))
        paths.append(f'<polyline points="{pts}" fill="none" stroke="#138a72" stroke-width="2" opacity="0.8"/>')
    static_circles = []
    for p in dataset.static_obstacles[scenario_id][:: max(1, len(dataset.static_obstacles[scenario_id]) // 250)]:
        x, y = xy(p)
        static_circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="#6f7782" opacity="0.42"/>')
    dyn_circles = []
    for p in dynamic:
        x, y = xy(p)
        dyn_circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#d14d3f" opacity="0.72"/>')
    markers = []
    for task in tasks:
        sx, sy = xy(task.start)
        gx, gy = xy(task.goal)
        markers.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="#225ea8"/><rect x="{gx-5:.1f}" y="{gy-5:.1f}" width="10" height="10" fill="#111827"/>')
    content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f8faf8"/>
<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="22" fill="#172026">Adaptive Feedback Graph, Scenario {scenario_id} ({scenario.dynamic})</text>
<rect x="40" y="40" width="{width - 80}" height="{height - 80}" fill="#ffffff" stroke="#172026"/>
<g>{''.join(static_circles)}</g>
<g>{''.join(dyn_circles)}</g>
<g>{''.join(paths)}</g>
<g>{''.join(markers)}</g>
<text x="52" y="{height - 18}" font-family="Arial" font-size="13" fill="#172026">blue = UAV start, black = assigned task, gray = static obstacles, red = live dynamic obstacles, green = adaptive paths</text>
</svg>"""
    path.write_text(content, encoding="utf-8")


def write_summary(path: Path, rows: Sequence[PlanMetrics]) -> None:
    by_method: Dict[str, List[PlanMetrics]] = defaultdict(list)
    for row in rows:
        by_method[row.method].append(row)
    summary = {}
    for method, values in by_method.items():
        summary[method] = {
            "objective_mean": sum(v.objective() for v in values) / len(values),
            "collision_risk_mean": sum(v.collision_risk for v in values) / len(values),
            "makespan_mean": sum(v.makespan for v in values) / len(values),
            "energy_used_mean": sum(v.energy_used for v in values) / len(values),
            "completed_mean": sum(v.completed for v in values) / len(values),
            "graph_updates_total": sum(v.graph_updates for v in values),
        }
    base = summary.get("static_graph", {}).get("objective_mean")
    central = summary.get("central_feedback_static_topology", {}).get("objective_mean")
    adaptive = summary.get("adaptive_feedback_hgrl", {}).get("objective_mean")
    if base and adaptive:
        summary["adaptive_vs_static_objective_improvement_percent"] = 100.0 * (base - adaptive) / base
    if central and adaptive:
        summary["adaptive_vs_central_feedback_objective_improvement_percent"] = 100.0 * (central - adaptive) / central
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    dataset = UAVDataset(Path(args.dataset))
    scenario_ids = sorted(dataset.scenarios)
    if args.scenarios:
        wanted = {int(s) for s in args.scenarios.split(",")}
        scenario_ids = [sid for sid in scenario_ids if sid in wanted]
    methods = ["static_graph", "central_feedback_static_topology", "adaptive_feedback_hgrl"]
    rows: List[PlanMetrics] = []
    for sid in scenario_ids:
        scenario = dataset.scenarios[sid]
        for method in methods:
            rows.append(simulate_method(dataset, scenario, method, args.resolution, args.seed, args.max_agents))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "metrics.csv", rows)
    write_summary(out / "summary.json", rows)
    svg_bar_chart(out / "objective_comparison.svg", "Lower Multi-Objective Score Is Better", rows, "objective")
    svg_bar_chart(out / "risk_comparison.svg", "Lower Collision Risk Is Better", rows, "collision_risk")
    dynamic_ids = [sid for sid in scenario_ids if dataset.scenarios[sid].dynamic == "dynamic"]
    svg_scenario_map(out / "adaptive_scenario_map.svg", dataset, dynamic_ids[0] if dynamic_ids else scenario_ids[0], args.resolution)
    print(f"Wrote results to {out.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive feedback-driven hierarchical graph planner experiment")
    parser.add_argument("--dataset", default="data_raw/uav_dataset/uav_dataset")
    parser.add_argument("--out", default="outputs/adaptive_hgrl")
    parser.add_argument("--resolution", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-agents", type=int, default=8)
    parser.add_argument("--scenarios", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
