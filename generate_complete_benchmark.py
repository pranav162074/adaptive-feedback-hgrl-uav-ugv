from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Iterable, List, Tuple


Point = Tuple[float, float]


def clamp(value: float) -> float:
    return max(0.02, min(0.98, value))


def write_csv(path: Path, fields: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clustered_points(rng: random.Random, centers: List[Point], count: int, spread: float) -> List[Point]:
    points: List[Point] = []
    for i in range(count):
        cx, cy = centers[i % len(centers)]
        points.append((clamp(rng.gauss(cx, spread)), clamp(rng.gauss(cy, spread))))
    return points


def generate(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    out = Path(args.out)
    scenarios = []
    agents = []
    tasks = []
    static_obstacles = []
    dynamic_obstacles = []
    terrain_cost = []
    communication_events = []
    battery_events = []
    feedback_events = []
    uav_positions = []

    for sid in range(args.scenarios):
        density = [0.12, 0.22, 0.34, 0.46][sid % 4]
        dynamic = "dynamic" if sid % 2 else "static"
        clustered = "clustered" if sid % 3 != 1 else "dispersed"
        scenarios.append(
            {
                "scenario_id": sid,
                "density": density,
                "clustered": clustered,
                "dynamic": dynamic,
                "map_width": 1.0,
                "map_height": 1.0,
                "mission_horizon": args.horizon,
                "description": f"{clustered}_{dynamic}_battery_comm_terrain_case_{sid}",
            }
        )

        ugv_count = 2 + sid % 2
        uav_count = args.uavs
        support_points: List[Point] = []
        for gid in range(ugv_count):
            x = 0.08 + 0.08 * gid
            y = 0.10 + 0.12 * (gid % 2)
            support_points.append((x, y))
            agents.append(
                {
                    "scenario_id": sid,
                    "agent_id": f"UGV_{gid}",
                    "type": "UGV",
                    "start_x": x,
                    "start_y": y,
                    "speed": 0.55 + 0.04 * gid,
                    "battery_capacity": 320,
                    "battery_start": 300 - 12 * gid,
                    "energy_rate": 0.20,
                    "comm_range": 0.48,
                    "payload_capacity": 8 + gid,
                    "sensor_range": 0.18,
                    "can_recharge": "true",
                    "recharge_rate": 4.0,
                }
            )

        for uid in range(uav_count):
            sx = 0.05 + 0.09 * (uid % 4)
            sy = 0.86 - 0.10 * (uid // 4)
            battery_start = 64 - 7 * (uid % 5) - 4 * (sid % 3)
            if uid in (2, 5):
                battery_start -= 12
            agents.append(
                {
                    "scenario_id": sid,
                    "agent_id": f"UAV_{uid}",
                    "type": "UAV",
                    "start_x": sx,
                    "start_y": sy,
                    "speed": 1.15 + 0.08 * (uid % 3),
                    "battery_capacity": 100,
                    "battery_start": battery_start,
                    "energy_rate": 0.75 + 0.06 * (uid % 4),
                    "comm_range": 0.34 - 0.02 * (uid % 2),
                    "payload_capacity": 2 + (uid % 3),
                    "sensor_range": 0.12 + 0.02 * (uid % 2),
                    "can_recharge": "false",
                    "recharge_rate": 0.0,
                }
            )

        task_centers = [(0.78, 0.76), (0.62, 0.32), (0.42, 0.66), (0.86, 0.44)]
        for tid in range(args.tasks):
            tx, ty = clustered_points(rng, task_centers, 1, 0.08 + 0.01 * sid)[0]
            priority = 1 + ((tid + sid) % 5)
            deadline = 35 + 8 * tid + 4 * sid
            requires_ugv = "true" if tid % 5 == 0 else "false"
            tasks.append(
                {
                    "scenario_id": sid,
                    "task_id": f"T_{tid}",
                    "x": tx,
                    "y": ty,
                    "priority": priority,
                    "deadline": deadline,
                    "service_time": 2 + tid % 4,
                    "payload_required": 1 + tid % 3,
                    "requires_uav": "true",
                    "requires_ugv": requires_ugv,
                    "risk_sensitivity": round(0.45 + 0.1 * (priority % 3), 3),
                }
            )
            if tid < uav_count:
                sx = 0.05 + 0.09 * (tid % 4)
                sy = 0.86 - 0.10 * (tid // 4)
                uav_positions.append(
                    {
                        "scenario_id": sid,
                        "uav_id": tid,
                        "start_x": sx,
                        "start_y": sy,
                        "goal_x": tx,
                        "goal_y": ty,
                    }
                )

        obstacle_centers = [(0.38, 0.45), (0.55, 0.58), (0.70, 0.25)] if clustered == "clustered" else [(rng.random(), rng.random()) for _ in range(6)]
        static_points = clustered_points(rng, obstacle_centers, int(240 * density), 0.055 if clustered == "clustered" else 0.18)
        for oid, (x, y) in enumerate(static_points):
            static_obstacles.append({"scenario_id": sid, "obstacle_id": oid, "x": x, "y": y, "radius": 0.018, "severity": 0.75})

        for cell_id in range(args.terrain_cells):
            gx = cell_id % int(math.sqrt(args.terrain_cells))
            gy = cell_id // int(math.sqrt(args.terrain_cells))
            x = (gx + 0.5) / int(math.sqrt(args.terrain_cells))
            y = (gy + 0.5) / int(math.sqrt(args.terrain_cells))
            roughness = 1.0 + 0.55 * math.exp(-8 * math.dist((x, y), (0.52, 0.50))) + 0.15 * rng.random()
            terrain_cost.append({"scenario_id": sid, "cell_id": cell_id, "x": x, "y": y, "terrain_cost": round(roughness, 4)})

        moving_count = 8 if dynamic == "dynamic" else 2
        for oid in range(moving_count):
            base_x = 0.22 + 0.08 * oid
            base_y = 0.18 + 0.07 * (oid % 5)
            for t in range(args.horizon):
                x = clamp(base_x + 0.12 * math.sin(0.09 * t + oid + sid))
                y = clamp(base_y + 0.10 * math.cos(0.07 * t + 0.5 * oid))
                dynamic_obstacles.append(
                    {
                        "scenario_id": sid,
                        "n_dynamic_obstacles": moving_count,
                        "obstacle_id": oid,
                        "time_step": t,
                        "x": x,
                        "y": y,
                        "theta": round(math.atan2(y - base_y, x - base_x), 5),
                        "radius": 0.032 + 0.004 * (oid % 3),
                        "severity": 0.65 + 0.05 * (oid % 4),
                    }
                )

        for t in [8, 16, 24, 32, 48, 64]:
            communication_events.append(
                {
                    "scenario_id": sid,
                    "time_step": t,
                    "event_type": "communication_degradation",
                    "x": clamp(0.34 + 0.18 * math.sin((sid + t) / 7)),
                    "y": clamp(0.54 + 0.14 * math.cos((sid + t) / 8)),
                    "radius": 0.16,
                    "severity": 0.55 + 0.1 * (sid % 3),
                }
            )
            feedback_events.append(
                {
                    "scenario_id": sid,
                    "time_step": t,
                    "event_type": "risk_zone",
                    "target_id": "",
                    "x": clamp(0.48 + 0.18 * math.sin((sid + t) / 6)),
                    "y": clamp(0.48 + 0.18 * math.cos((sid + t) / 7)),
                    "radius": 0.18,
                    "severity": 0.82,
                }
            )

        for uid in range(uav_count):
            for t in [16, 24, 32]:
                battery_events.append(
                    {
                        "scenario_id": sid,
                        "time_step": t,
                        "agent_id": f"UAV_{uid}",
                        "event_type": "battery_warning",
                        "battery_delta": -10 - 2 * (uid % 3),
                        "severity": 0.75,
                    }
                )

    write_csv(out / "scenarios.csv", list(scenarios[0]), scenarios)
    write_csv(out / "agents.csv", list(agents[0]), agents)
    write_csv(out / "tasks.csv", list(tasks[0]), tasks)
    write_csv(out / "uav_positions.csv", list(uav_positions[0]), uav_positions)
    write_csv(out / "static_obstacles.csv", list(static_obstacles[0]), static_obstacles)
    write_csv(out / "dynamic_obstacles.csv", list(dynamic_obstacles[0]), dynamic_obstacles)
    write_csv(out / "terrain_cost.csv", list(terrain_cost[0]), terrain_cost)
    write_csv(out / "communication_events.csv", list(communication_events[0]), communication_events)
    write_csv(out / "battery_events.csv", list(battery_events[0]), battery_events)
    write_csv(out / "feedback_events.csv", list(feedback_events[0]), feedback_events)

    readme = """# Complete UAV/UGV Adaptive Graph Benchmark

Synthetic benchmark designed for centralized feedback-driven graph reinforcement learning.

It includes UAV and UGV agents, heterogeneous capabilities, battery profiles, dynamic obstacles,
terrain costs, communication degradation events, explicit feedback events, task deadlines,
task priorities, and UAV/UGV support requirements.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote complete benchmark dataset to {out.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate complete UAV/UGV adaptive graph benchmark dataset")
    parser.add_argument("--out", default="data_raw/complete_adaptive_benchmark")
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--uavs", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--terrain-cells", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
