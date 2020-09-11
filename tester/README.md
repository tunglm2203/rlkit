Instruction for data generation:

# Pairwise data

[1] Generate random-pairwise trajectories: store image of `observation`\
Source: `SawyerPushNIPSEasy-v0`\
Target: `SawyerPushXYReal-v0`\
Files:
- `gen_rand_pair_im_sim.py`: Run first, file `random_trajectories.npz` contain simulator's trajectories
- `gen_rand_pair_im_real.py`: Run after

[2] Generate random-pairwise trajectories: store image of `observation`\
Source: `SawyerPushNIPSEasy-v0`\
Target: `SawyerPushXYReal-v0`\
Files:
- `gen_rand_pair_im_sim_1.py`: Run after
- `gen_rand_pair_im_real_1.py`: Run first, file `random_trajectories.npz` contain real's trajectories

[3] Generate random-pairwise trajectories: store image of `observation`\
Source: `SawyerPushNIPSEasy-v0`\
Target: `SawyerPushNIPSCustomEasy-v0`\
Files:
- `gen_rand_pair_im_src_2.py`: Run after
- `gen_rand_pair_im_tgt_2.py`: Run first, file `random_trajectories.npz` contain target's trajectories

[4] Generate random-pairwise goals: store image of `desired_goal`\
Source: `SawyerPushNIPSEasy-v0`\
Target: `SawyerPushNIPSCustomEasy-v0`\
Files:
- `gen_rand_pair_im_src_3.py`: Run after
- `gen_rand_pair_im_tgt_3.py`: Run first, file `random_trajectories.npz` contain target's trajectories

[5] Generate random-pairwise goals: store image of `desired_goal`\
Source: `SawyerPushNIPSEasy-v0`\
Target: `SawyerPushXYReal-v0`\
Files:
- `gen_rand_pair_im_src_4.py`: Run after
- `gen_rand_pair_im_tgt_4.py`: Run first, file `random_trajectories.npz` contain target's trajectories