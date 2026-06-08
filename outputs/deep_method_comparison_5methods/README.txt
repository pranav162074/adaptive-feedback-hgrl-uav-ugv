Five-Method Deep Comparison Outputs
==================================

Dataset:
data_raw/complete_adaptive_benchmark

Methods:
1. paper1_deep_hgrl_ugv_assisted
   Deep heterogeneous graph RL baseline inspired by the UGV-assisted heterogeneous graph reinforcement learning paper.

2. paper2_deep_cfr_marl
   Deep centralized-feedback MARL baseline inspired by the CFR-MARL paper.

3. paper3_deep_energy_uav_ugv_drl
   Deep energy-constrained UAV/UGV routing baseline inspired by recent UAV-UGV cooperative routing DRL work.

4. paper4_deep_tanet_td3_multi_uav
   Deep TANet-TD3-inspired multi-UAV target assignment and path-planning baseline.

5. proposed_adaptive_hgrl
   Our proposed adaptive feedback-driven hierarchical graph method.

Main comparison file:
comparison_numeric/comparison_summary.txt

Key result:
The proposed method has the lowest mean objective score and highest task completion.

Objective means:
paper1_deep_hgrl_ugv_assisted: 126.7674
paper2_deep_cfr_marl: 927.1795
paper3_deep_energy_uav_ugv_drl: 307.0818
paper4_deep_tanet_td3_multi_uav: 927.1036
proposed_adaptive_hgrl: 114.9820

Proposed improvement:
vs paper1_deep_hgrl_ugv_assisted: 9.2969%
vs paper2_deep_cfr_marl: 87.5987%
vs paper3_deep_energy_uav_ugv_drl: 62.5566%
vs paper4_deep_tanet_td3_multi_uav: 87.5977%
