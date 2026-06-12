Five-Method Deep Comparison Outputs
==================================

Dataset:
data_raw/complete_adaptive_benchmark

Scenario count:
100 heterogeneous UAV/UGV mission scenarios

Methods:
1. paper1_deep_hgrl_ugv_assisted
   Deep heterogeneous graph RL baseline inspired by the UGV-assisted heterogeneous graph reinforcement learning paper.

2. paper2_deep_cfr_marl
   Deep centralized-feedback MARL baseline inspired by the CFR-MARL paper.

3. paper3_deep_mw_maddpg_uav_swarm
   Deep MW-MADDPG UAV-swarm decision-making baseline inspired by the published Frontiers in Neurorobotics paper.

4. paper4_deep_tanet_td3_multi_uav
   Deep TANet-TD3-inspired multi-UAV target assignment and path-planning baseline.

5. proposed_adaptive_hgrl
   Our proposed adaptive feedback-driven hierarchical graph method.

Main comparison file:
comparison_numeric/comparison_summary.txt

Key result:
The proposed method has the lowest mean objective score while maintaining competitive task completion across the 100-scenario benchmark.

Objective means:
paper1_deep_hgrl_ugv_assisted: 80.1398
paper2_deep_cfr_marl: 83.3864
paper3_deep_mw_maddpg_uav_swarm: 83.7728
paper4_deep_tanet_td3_multi_uav: 83.6014
proposed_adaptive_hgrl: 79.6811

Proposed improvement:
vs paper1_deep_hgrl_ugv_assisted: 0.5724%
vs paper2_deep_cfr_marl: 4.4436%
vs paper3_deep_mw_maddpg_uav_swarm: 4.8843%
vs paper4_deep_tanet_td3_multi_uav: 4.6893%

Research graph data:
research_graph_data/

Research graph figures:
research_graphs/
