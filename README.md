# Adaptive Feedback-Driven Hierarchical Graph RL Prototype

This workspace contains a runnable prototype for the internship project:

> A Centralized Feedback-Driven Graph Reinforcement Learning Framework for Cooperative Path Planning and Task Allocation in Heterogeneous Environment

The implementation focuses on the proposed improvement over the two reference directions:

- Static heterogeneous graph task allocation: represented here by `static_graph`.
- Centralized feedback with static routing topology: represented here by `central_feedback_static_topology`.
- Proposed method: `adaptive_feedback_hgrl`.

The prototype uses the provided `uav_dataset` CSV files, including UAV starts/goals, static obstacles, dynamic obstacles, and scenario metadata.

## What Is Implemented

This repository now has two implementation layers:

- `adaptive_hgrl.py`: the best-performing adaptive feedback-driven hierarchical graph planner used for baseline comparison and graphs.
- `centralized_graph_rl.py`: an explicit centralized graph reinforcement learning trainer using tabular Q-learning, temporal-difference updates, centralized feedback rewards, and adaptive graph edge updates.

`adaptive_hgrl.py` implements a hierarchical planner:

1. High-level task allocation policy:
   - Assigns goals to UAVs using graph-cost estimates.
   - Includes a simple heterogeneous-agent bonus so agents are not treated as identical.

2. Low-level path planning policy:
   - Runs A* over a discretized graph.
   - Uses obstacle-aware edge weights.
   - Replans during execution when live feedback arrives.

3. Adaptive feedback graph:
   - Static obstacle risk is precomputed into the graph.
   - Dynamic obstacle feedback updates edge penalties.
   - Risky edges can become temporarily blocked, forcing the next replan to use safer alternatives.

4. Multi-objective evaluation:
   - Path cost.
   - Makespan.
   - Collision risk.
   - Energy use.
   - Completion count.
   - Fairness/load balance.

This is a practical, defensible prototype rather than a full deep-RL training pipeline. It preserves the research idea: centralized feedback modifies the graph online, while allocation and routing remain decoupled into hierarchical decisions.

`centralized_graph_rl.py` adds the reinforcement-learning component:

1. State:
   - Current graph node.
   - Goal node.
   - Dynamic-obstacle risk bucket.
   - Battery bucket.

2. Actions:
   - Movement to neighboring graph nodes.

3. Centralized feedback reward:
   - Positive reward for progress toward the assigned goal.
   - Penalty for path cost, dynamic obstacle risk, and battery use.
   - Terminal bonus for reaching the task goal.

4. Learning:
   - Q-values are updated using temporal-difference learning.
   - Risky traversed edges update the adaptive graph feedback penalties.
   - The trained value function is used for task assignment and path rollout.

This means the repository does contain an actual Centralized Feedback-Driven Graph Reinforcement Learning component. The current RL learner is intentionally lightweight and dependency-free. For publication-grade deep GRL, the tabular Q-function can be replaced with a GNN actor-critic while keeping the same dataset loader, feedback reward design, adaptive graph update logic, and evaluation metrics.

## Run

Use the bundled Codex Python runtime if your system Python does not have scientific packages:

```powershell
& 'C:\Users\Pranav\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\adaptive_hgrl.py
```

Optional dynamic-only run:

```powershell
& 'C:\Users\Pranav\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\adaptive_hgrl.py --scenarios 3,6,8
```

Run the explicit centralized graph RL trainer:

```powershell
python .\centralized_graph_rl.py
```

Faster dynamic-only RL smoke test:

```powershell
python .\centralized_graph_rl.py --scenarios 3,6,8 --episodes 120 --resolution 16
```

Outputs are written to:

```text
outputs/adaptive_hgrl/
```

## Current Results

Latest full run across all nine provided scenarios:

```json
{
  "static_graph": {
    "objective_mean": 79.2611,
    "collision_risk_mean": 0.0609
  },
  "central_feedback_static_topology": {
    "objective_mean": 76.4535,
    "collision_risk_mean": 0.0709
  },
  "adaptive_feedback_hgrl": {
    "objective_mean": 73.5393,
    "collision_risk_mean": 0.0255
  }
}
```

The adaptive method improves the mean multi-objective score by about 7.2% over the static graph baseline and reduces mean collision risk by about 58.1%.

Generated files:

- `metrics.csv`: per-scenario metrics for each method.
- `summary.json`: aggregate comparison.
- `objective_comparison.svg`: graph-ready objective comparison.
- `risk_comparison.svg`: graph-ready collision-risk comparison.
- `adaptive_scenario_map.svg`: example adaptive graph/path visualization.
- `outputs/centralized_graph_rl/rl_metrics.csv`: trained RL evaluation metrics.
- `outputs/centralized_graph_rl/rl_summary.json`: aggregate RL summary.

## Our Method
The proposed Adaptive Feedback-Driven Hierarchical Graph Reinforcement Learning framework separates task allocation and path planning into two hierarchical policies. The high-level policy assigns tasks based on estimated graph traversal cost and heterogeneous agent capability. The low-level policy performs feedback-aware path planning over an adaptive graph whose edge weights are updated using real-time obstacle and mission-state feedback. Unlike static graph approaches, the proposed method changes the graph during execution by penalizing or blocking risky edges, enabling safer replanning under dynamic environmental changes.

For a stronger final paper implementation, this prototype can be extended by replacing the tabular Q-learning policy with a trainable GNN actor-critic. The experiment harness, metrics, centralized feedback reward, and adaptive graph update logic can remain the same.
