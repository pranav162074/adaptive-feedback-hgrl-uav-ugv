from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt


METHODS = [
    "paper1_deep_hgrl_ugv_assisted",
    "paper2_deep_cfr_marl",
    "paper3_deep_mw_maddpg_uav_swarm",
    "paper4_deep_tanet_td3_multi_uav",
    "proposed_adaptive_hgrl",
]

LABELS = {
    "paper1_deep_hgrl_ugv_assisted": "HGRL-UGV",
    "paper2_deep_cfr_marl": "CFR-MARL",
    "paper3_deep_mw_maddpg_uav_swarm": "MW-MADDPG",
    "paper4_deep_tanet_td3_multi_uav": "TANet-TD3",
    "proposed_adaptive_hgrl": "AHG-FMARL",
}


def load_metrics(root: Path) -> dict[str, list[dict]]:
    metrics: dict[str, list[dict]] = {}
    for method in METHODS:
        payload = json.loads((root / method / "metrics.json").read_text(encoding="utf-8"))
        metrics[method] = payload["per_episode"]
    return metrics


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def fmt(mean: float, std: float, decimals: int = 2) -> str:
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def table_rows(metrics: dict[str, list[dict]], columns: list[tuple[str, Callable[[dict], float], int]]) -> list[dict]:
    rows = []
    for method in METHODS:
        episodes = metrics[method]
        row = {"Method": LABELS[method]}
        for name, getter, decimals in columns:
            values = [getter(ep) for ep in episodes]
            mean, std = mean_std(values)
            row[name] = fmt(mean, std, decimals)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_table(path: Path, title: str, rows: list[dict]) -> None:
    headers = list(rows[0])
    cell_text = [[row[h] for h in headers] for row in rows]
    fig_width = max(8.0, 1.35 * len(headers))
    fig_height = 1.35 + 0.48 * len(rows)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=220)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)

    proposed_index = next(i for i, row in enumerate(rows) if row["Method"] == "AHG-FMARL")
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")
        if r == proposed_index + 1:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eaf5ec")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance comparison tables from 40k outputs")
    parser.add_argument("--root", default="outputs/deep_method_comparison_5methods")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / "performance_tables"
    metrics = load_metrics(root)

    mission_columns = [
        ("Obj. score ↓", lambda ep: ep["objective"], 2),
        ("TC (%) ↑", lambda ep: 100.0 * ep["completed"] / 8.0, 2),
        ("TCR ↓", lambda ep: ep["makespan"], 3),
        ("CR (%) ↓", lambda ep: 100.0 * ep["collision_risk"], 2),
        ("Energy ↓", lambda ep: ep["energy_used"], 3),
        ("AR ↑", lambda ep: 1.0 / max(1e-9, ep["objective"]) * 10000.0, 2),
    ]
    resource_columns = [
        ("Path len. ↓", lambda ep: ep["avg_path_length"], 3),
        ("Min battery ↑", lambda ep: ep["min_battery"], 3),
        ("Recharge visits", lambda ep: ep["recharge_visits"], 2),
        ("Batt. warnings", lambda ep: ep["battery_warnings"], 2),
        ("Graph updates ↑", lambda ep: ep["graph_updates"], 2),
        ("Fairness std ↓", lambda ep: ep["fairness_std"], 3),
    ]
    normalized_columns = [
        ("Completion rate ↑", lambda ep: ep["completed"] / 8.0, 3),
        ("Time index ↓", lambda ep: ep["makespan"] / 2.5, 3),
        ("Energy index ↓", lambda ep: ep["energy_used"] / 7.0, 3),
        ("Risk index ↓", lambda ep: ep["collision_risk"], 3),
        ("Battery index ↑", lambda ep: ep["min_battery"] / 0.30, 3),
        ("Objective index ↓", lambda ep: ep["objective"] / 100.0, 3),
    ]

    table_specs = [
        ("table_1_mission_performance", "Performance Metrics Comparison", mission_columns),
        ("table_2_resource_battery_feedback", "Resource, Battery, and Feedback Metrics Comparison", resource_columns),
        ("table_3_normalized_indices", "Normalized Performance Indices Comparison", normalized_columns),
    ]

    for filename, title, columns in table_specs:
        rows = table_rows(metrics, columns)
        write_csv(out_dir / f"{filename}.csv", rows)
        render_table(out_dir / filename, title, rows)

    definitions = {
        "Obj. score": "multi-objective mission cost; lower is better",
        "TC (%)": "completed tasks / assigned tasks * 100; higher is better",
        "TCR": "task completion time ratio represented by makespan in normalized mission time; lower is better",
        "CR (%)": "collision/path risk percentage; lower is better",
        "Energy": "mission energy used; lower is better",
        "AR": "aggregate reward-style score computed as 10000/objective; higher is better",
        "Path len.": "average path length",
        "Min battery": "minimum remaining battery ratio",
        "Graph updates": "adaptive graph feedback updates; higher indicates active feedback adaptation",
        "Fairness std": "load-balance/fairness standard deviation; lower is better",
        "source": "All table values are mean +/- standard deviation over the 100 per-episode records from the 40,000-training-episode Kaggle run.",
    }
    (out_dir / "metric_definitions.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    print(f"Wrote performance tables to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
