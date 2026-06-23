# Adaptive Feedback HGRL UAV/UGV Research Implementation

Project:

> A Centralized Feedback-Driven Graph Reinforcement Learning Framework for Cooperative Path Planning and Task Allocation in Heterogeneous Environment

Proposed method:

> AHG-FMARL: Adaptive Feedback-Driven Hierarchical Graph Reinforcement Learning for cooperative UAV/UGV task allocation and path planning.

This project compares four published-method-inspired deep graph/RL baselines against the proposed adaptive feedback method:

- `paper1_deep_hgrl_ugv_assisted`: UGV-assisted heterogeneous graph RL inspired baseline.
- `paper2_deep_cfr_marl`: centralized feedback/reward MARL inspired baseline.
- `paper3_deep_mw_maddpg_uav_swarm`: MW-MADDPG UAV-swarm inspired baseline.
- `paper4_deep_tanet_td3_multi_uav`: TANet-TD3 multi-UAV target assignment/path-planning inspired baseline.
- `proposed_adaptive_hgrl`: proposed AHG-FMARL method.

## Current Workflow

The current project is organized around this pipeline:

```text
generate_complete_benchmark.py
    -> data_raw/complete_adaptive_benchmark/
    -> run_deep_existing_methods.py
    -> outputs/deep_method_comparison_5methods/
    -> research_graphs/
```

## Benchmark Dataset

The active benchmark is stored in:

```text
data_raw/complete_adaptive_benchmark/
```

The current local benchmark contains `40,000` generated episodes named:

```text
E1 ... E40000
```

Main dataset files:

- `episodes.csv`: episode metadata, mission horizon, communication range, observation range, difficulty index, and UAV/UGV/PoI counts.
- `agents.csv`: UAV/UGV type, position, speed, battery, payload, communication, sensor, recharge, and processing capability.
- `tasks.csv`: task/PoI position, priority, deadline, payload, data volume, compute demand, and UAV/UGV support requirement.
- `static_obstacles.csv`: static obstacle field.
- `dynamic_obstacles.csv`: moving obstacle trajectories.
- `terrain_cost.csv`: terrain-cost grid.
- `communication_events.csv`: communication degradation feedback.
- `battery_events.csv`: battery warning/drop events.
- `feedback_events.csv`: centralized risk feedback events.
- `uav_positions.csv`: compatibility mapping for UAV start/goal style loaders.

Regenerate the full benchmark:

```powershell
python .\generate_complete_benchmark.py --episodes 40000
```

## Main 5-Method Runner

Install dependencies on Kaggle:

```bash
pip install -r requirements-kaggle.txt
```

Run the full 40,000-training-episode comparison:

```bash
python run_deep_existing_methods.py \
  --dataset data_raw/complete_adaptive_benchmark \
  --out outputs/deep_method_comparison_5methods \
  --training-episodes 40000 \
  --log-interval 1 \
  --device auto
```

For a tiny local check, if PyTorch is installed locally:

```powershell
python .\run_deep_existing_methods.py --training-episodes 2 --log-interval 1 --episode-filter E1,E2 --eval-episodes 2 --device cpu --out outputs/smoke_5methods_episode_upgrade
```

## Outputs

The latest output folder is:

```text
outputs/deep_method_comparison_5methods/
```

Each method folder contains:

- `training_history.csv`: full 40,000-episode logged training history.
- `metrics.json`: final sampled path-evaluation metrics.
- `summary.txt`: readable method summary.

The research graphs are stored in:

```text
outputs/deep_method_comparison_5methods/research_graphs/
```

Current graph files:

- `00_four_metric_episode_curves.png`
- `01_data_collection_rate_vs_episodes.png`
- `02_task_completion_rate_vs_episodes.png`
- `03_task_completion_time_ratio_vs_episodes.png`
- `04_energy_consumption_index_vs_episodes.png`
- `episode_curve_plot_data_400points.csv`

The graph data is generated from the five full `training_history.csv` files by taking every 100th episode, giving `400` plotted points per method.

## Current Result Interpretation

For the episode-wise graph metrics:

- Higher is better: data collection rate, task completion rate.
- Lower is better: task completion time ratio, energy consumption index, objective index.

The current research graphs show AHG-FMARL as the best method across the plotted episode-wise metrics.

## Lightweight Adaptive Planner

The standalone adaptive planner remains available without PyTorch:

```powershell
python .\adaptive_hgrl.py --dataset data_raw/complete_adaptive_benchmark --out outputs/adaptive_hgrl
```

Use `--write-visuals` only when SVG maps/plots are explicitly needed.
