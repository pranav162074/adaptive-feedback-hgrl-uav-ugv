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
    episodes = []
    agents = []
    tasks = []
    static_obstacles = []
    dynamic_obstacles = []
    terrain_cost = []
    communication_events = []
    battery_events = []
    feedback_events = []
    uav_positions = []

    for sid in range(args.episodes):
        episode_id = f"E{sid + 1}"
        phase = sid / max(1, args.episodes - 1)
        density = [0.10, 0.18, 0.28, 0.38, 0.48][sid % 5]
        dynamic = "dynamic" if sid % 3 != 0 else "mixed"
        clustered = "clustered" if sid % 4 in (0, 2, 3) else "dispersed"
        poi_count = args.min_tasks + (sid % (args.max_tasks - args.min_tasks + 1))
        uav_count = args.min_uavs + (sid % (args.max_uavs - args.min_uavs + 1))
        ugv_count = args.min_ugvs + (sid % (args.max_ugvs - args.min_ugvs + 1))
        observation_range_m = 70 + 10 * (sid % 7)
        communication_range_m = 110 + 15 * (sid % 9)
        bandwidth_mbps = round(8.0 + 8.0 * (1.0 - density) + 1.5 * math.sin(sid / 23.0), 3)
        latency_ms = round(18.0 + 36.0 * density + 5.0 * (sid % 4), 3)
        packet_loss_rate = round(0.015 + 0.11 * density + 0.015 * ((sid // 7) % 3), 4)
        episodes.append(
            {
                "episode_id": episode_id,
                "density": density,
                "clustered": clustered,
                "dynamic": dynamic,
                "map_width_m": 1000,
                "map_height_m": 1000,
                "mission_horizon_s": args.horizon,
                "poi_count": poi_count,
                "uav_count": uav_count,
                "ugv_count": ugv_count,
                "observation_range_m": observation_range_m,
                "communication_range_m": communication_range_m,
                "bandwidth_mbps": bandwidth_mbps,
                "latency_ms": latency_ms,
                "packet_loss_rate": packet_loss_rate,
                "difficulty_index": round(0.25 + 0.60 * density + 0.15 * phase + 0.05 * math.sin(sid / 17.0), 4),
                "description": f"{clustered}_{dynamic}_episode_{sid + 1}",
            }
        )

        support_points: List[Point] = []
        for gid in range(ugv_count):
            x = 0.08 + 0.08 * gid
            y = 0.10 + 0.12 * (gid % 2)
            support_points.append((x, y))
            agents.append(
                {
                    "episode_id": episode_id,
                    "agent_id": f"UGV_{gid}",
                    "type": "UGV",
                    "start_x": x,
                    "start_y": y,
                    "speed_mps": 8.0 + 0.5 * gid,
                    "speed": 0.55 + 0.04 * gid,
                    "battery_capacity_Wh": 320,
                    "battery_capacity": 320,
                    "battery_start_Wh": 300 - 12 * gid,
                    "battery_start": 300 - 12 * gid,
                    "energy_rate_Wh_per_m": 0.020,
                    "energy_rate": 0.20,
                    "comm_range_m": communication_range_m,
                    "comm_range": communication_range_m / 1000.0,
                    "payload_capacity": 8 + gid,
                    "sensor_range_m": observation_range_m,
                    "sensor_range": observation_range_m / 1000.0,
                    "can_recharge": "true",
                    "recharge_rate_Wh_per_s": 4.0,
                    "recharge_rate": 4.0,
                    "cpu_ghz": round(8.0 + 0.5 * gid, 2),
                    "bandwidth_mbps": bandwidth_mbps * 1.35,
                    "latency_ms": latency_ms * 0.8,
                    "processing_energy_wh_per_mbit": 0.018,
                    "communication_energy_wh_per_mbit": 0.018,
                }
            )

        for uid in range(uav_count):
            sx = 0.05 + 0.09 * (uid % 4)
            sy = 0.86 - 0.10 * (uid // 4)
            battery_start = 66 - 6 * (uid % 5) - 5 * (sid % 4)
            if uid in (2, 5):
                battery_start -= 12
            agents.append(
                {
                    "episode_id": episode_id,
                    "agent_id": f"UAV_{uid}",
                    "type": "UAV",
                    "start_x": sx,
                    "start_y": sy,
                    "speed_mps": 22.0 + 1.2 * (uid % 3),
                    "speed": 1.15 + 0.08 * (uid % 3),
                    "battery_capacity_Wh": 100,
                    "battery_capacity": 100,
                    "battery_start_Wh": battery_start,
                    "battery_start": battery_start,
                    "energy_rate_Wh_per_m": 0.075 + 0.006 * (uid % 4),
                    "energy_rate": 0.75 + 0.06 * (uid % 4),
                    "comm_range_m": communication_range_m * (0.82 - 0.04 * (uid % 2)),
                    "comm_range": (communication_range_m / 1000.0) * (0.82 - 0.04 * (uid % 2)),
                    "payload_capacity": 2 + (uid % 3),
                    "sensor_range_m": observation_range_m * (0.82 + 0.06 * (uid % 2)),
                    "sensor_range": (observation_range_m / 1000.0) * (0.82 + 0.06 * (uid % 2)),
                    "can_recharge": "false",
                    "recharge_rate_Wh_per_s": 0.0,
                    "recharge_rate": 0.0,
                    "cpu_ghz": round(3.5 + 0.45 * (uid % 4), 2),
                    "bandwidth_mbps": bandwidth_mbps * (0.85 - 0.03 * (uid % 3)),
                    "latency_ms": latency_ms * (1.05 + 0.04 * (uid % 2)),
                    "processing_energy_wh_per_mbit": 0.040 + 0.004 * (uid % 3),
                    "communication_energy_wh_per_mbit": 0.026 + 0.003 * (uid % 4),
                }
            )

        task_centers = [(0.78, 0.76), (0.62, 0.32), (0.42, 0.66), (0.86, 0.44)]
        for tid in range(poi_count):
            tx, ty = clustered_points(rng, task_centers, 1, 0.055 + 0.035 * density)[0]
            priority = 1 + ((tid + sid) % 5)
            deadline = 38 + 3.5 * tid + 18 * density + 4 * ((sid // 5) % 3)
            requires_ugv = "true" if tid % 4 == 0 or packet_loss_rate > 0.08 else "false"
            data_volume = round(0.8 + 14.2 * rng.random() + 2.0 * density + 0.25 * priority, 4)
            required_data = round(data_volume * (0.88 + 0.08 * rng.random()), 4)
            tasks.append(
                {
                    "episode_id": episode_id,
                    "poi_id": f"P_{tid}",
                    "task_id": f"P_{tid}",
                    "x": tx,
                    "y": ty,
                    "priority": priority,
                    "deadline_s": round(deadline, 3),
                    "deadline": round(deadline, 3),
                    "service_time": 2 + tid % 4,
                    "payload_required": 1 + tid % 3,
                    "requires_uav": "true",
                    "requires_ugv": requires_ugv,
                    "risk_sensitivity": round(0.45 + 0.1 * (priority % 3), 3),
                    "data_volume_mbit": data_volume,
                    "required_data_mbit": required_data,
                    "compute_cycles_per_bit": 650 + 50 * ((tid + sid) % 9),
                    "collection_deadline_s": round(deadline * (0.92 + 0.08 * rng.random()), 3),
                    "offload_required": "true" if requires_ugv == "true" or data_volume > 10.0 else "false",
                }
            )
            if tid < uav_count:
                sx = 0.05 + 0.09 * (tid % 4)
                sy = 0.86 - 0.10 * (tid // 4)
                uav_positions.append(
                    {
                        "episode_id": episode_id,
                        "uav_id": tid,
                        "start_x": sx,
                        "start_y": sy,
                        "goal_x": tx,
                        "goal_y": ty,
                    }
                )

        obstacle_centers = [(0.38, 0.45), (0.55, 0.58), (0.70, 0.25)] if clustered == "clustered" else [(rng.random(), rng.random()) for _ in range(6)]
        static_points = clustered_points(rng, obstacle_centers, int(args.static_obstacle_scale * density), 0.055 if clustered == "clustered" else 0.18)
        for oid, (x, y) in enumerate(static_points):
            static_obstacles.append({"episode_id": episode_id, "obstacle_id": oid, "x": x, "y": y, "radius": 0.018, "severity": 0.75})

        for cell_id in range(args.terrain_cells):
            gx = cell_id % int(math.sqrt(args.terrain_cells))
            gy = cell_id // int(math.sqrt(args.terrain_cells))
            x = (gx + 0.5) / int(math.sqrt(args.terrain_cells))
            y = (gy + 0.5) / int(math.sqrt(args.terrain_cells))
            roughness = 1.0 + 0.55 * math.exp(-8 * math.dist((x, y), (0.52, 0.50))) + 0.15 * rng.random()
            terrain_cost.append({"episode_id": episode_id, "cell_id": cell_id, "x": x, "y": y, "terrain_cost_index": round(roughness, 4), "terrain_cost": round(roughness, 4)})

        moving_count = 4 + sid % 4 if dynamic == "dynamic" else 2
        for oid in range(moving_count):
            base_x = 0.22 + 0.08 * oid
            base_y = 0.18 + 0.07 * (oid % 5)
            for t in [0, 8, 16, 24, 32, 48, 64, args.horizon - 1]:
                x = clamp(base_x + 0.12 * math.sin(0.09 * t + oid + sid))
                y = clamp(base_y + 0.10 * math.cos(0.07 * t + 0.5 * oid))
                dynamic_obstacles.append(
                    {
                        "episode_id": episode_id,
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
                    "episode_id": episode_id,
                    "time_step": t,
                    "event_type": "communication_degradation",
                    "x": clamp(0.34 + 0.18 * math.sin((sid + t) / 7)),
                    "y": clamp(0.54 + 0.14 * math.cos((sid + t) / 8)),
                    "radius": 0.16,
                    "severity": 0.55 + 0.1 * (sid % 3),
                    "bandwidth_multiplier": round(max(0.35, 1.0 - 0.55 * density - 0.05 * (sid % 3)), 4),
                    "latency_ms": round(latency_ms * (1.4 + 0.15 * (sid % 4)), 3),
                    "packet_loss_rate": round(min(0.30, packet_loss_rate + 0.04 * (sid % 3)), 4),
                }
            )
            feedback_events.append(
                {
                    "episode_id": episode_id,
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
                        "episode_id": episode_id,
                        "time_step": t,
                        "agent_id": f"UAV_{uid}",
                        "event_type": "battery_warning",
                        "battery_delta": -10 - 2 * (uid % 3),
                        "battery_delta_Wh": -10 - 2 * (uid % 3),
                        "severity": 0.75,
                    }
                )

    write_csv(out / "episodes.csv", list(episodes[0]), episodes)
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
    parser.add_argument("--episodes", type=int, default=40000)
    parser.add_argument("--min-uavs", type=int, default=5)
    parser.add_argument("--max-uavs", type=int, default=25)
    parser.add_argument("--min-ugvs", type=int, default=1)
    parser.add_argument("--max-ugvs", type=int, default=8)
    parser.add_argument("--min-tasks", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--terrain-cells", type=int, default=16)
    parser.add_argument("--static-obstacle-scale", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
