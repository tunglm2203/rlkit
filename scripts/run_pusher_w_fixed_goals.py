import argparse
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from rlkit.core import logger
from rlkit.core.logging import MyEncoder
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv

import torch


def simulate_policy(args):
    # ===================== LOGGER SCOPE =====================
    # Check path of stored directory
    if not os.path.exists(args.result_path):
        print('[ERROR-AIM] The directory to store result is not exist: ', args.result_path)
        return
    if args.exp == '':
        print('[WARNING-AIM] You should set name of experiment.')
    saved_dir = os.path.join(args.result_path, args.exp)
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    res_file = os.path.join(saved_dir, 'test_policy.npz')
    paths_file = os.path.join(saved_dir, 'episodes.npz')

    with open(os.path.join(saved_dir, 'log_params.json'), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, cls=MyEncoder)

    # ===================== LOAD & CREATE MODEL SCOPE =====================
    # Load necessary data
    data = torch.load(open(args.file, "rb"))
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    print("Policy and environment loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
        # policy.to(ptu.device)
    if isinstance(env, VAEWrappedEnv) and hasattr(env, 'mode'):
        env.mode(args.mode)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()

    # ===================== SETUP ENVIRONMENT SCOPE =====================
    # TUNG:
    env._goal_sampling_mode = 'reset_of_env'
    env.wrapped_env.wrapped_env.randomize_goals = False

    # Set goal to set of fixed goals
    set_of_goals = np.array([
        [[0.55, 0.], [0.65, 0.1]],
        [[0.55, -0.1], [0.7, 0.1]],
        [[0.65, -0.1], [0.7, 0.05]],
        [[0.55, -0.1], [0.7, 0.05]],
        [[0.60, 0.], [0.7, 0.05]],
        [[0.55, 0.1], [0.7, -0.05]],
        [[0.55, 0.05], [0.55, -0.05]],
        [[0.60, 0.1], [0.7, -0.05]],
        [[0.65, -0.1], [0.65, 0.1]],
        [[0.55, -0.05], [0.7, -0.05]],
        [[0.55, 0.1], [0.7, 0.1]],
        [[0.60, 0.1], [0.7, 0.15]],
        [[0.60, 0.05], [0.7, 0.05]],
        [[0.60, -0.05], [0.55, 0.05]],
        [[0.55, -0.05], [0.7, 0.05]],
        [[0.60, 0.], [0.55, -0.1]],
        [[0.55, -0.1], [0.7, -0.05]],
        [[0.60, 0.], [0.7, -0.05]],
        [[0.60, -0.05], [0.7, -0.05]],
        [[0.55, -0.1], [0.5, -0.15]],
        [[0.60, 0.05], [0.65, 0.15]],
        [[0.55, -0.05], [0.55, 0.1]],
        [[0.55, 0.05], [0.60, 0.15]],
        [[0.65, 0.], [0.7, 0.1]],
        [[0.60, 0.05], [0.7, 0.15]],
        [[0.60, -0.05], [0.55, 0.1]],
        [[0.60, 0.05], [0.7, -0.1]],
        [[0.55, -0.05], [0.5, -0.15]],
        [[0.65, 0.05], [0.7, -0.05]],
        [[0.6, -0.05], [0.5, -0.1]]
    ])
    n_goals = len(set_of_goals)

    # ===================== TESTING SCOPE =====================
    # This piece of code for test learn policy on REAL env
    paths, puck_distances, hand_distances, goals = [], [], [], []
    print('[INFO] Starting test in set of goals...')
    for goal_id in tqdm(range(n_goals)):
        # Assign goal from set of goals
        env.wrapped_env.wrapped_env.fixed_hand_goal = set_of_goals[goal_id][0]
        env.wrapped_env.wrapped_env.fixed_puck_goal = set_of_goals[goal_id][1]

        # Number of test per each goal
        paths_per_goal = []
        for i in range(args.n_test):
            paths_per_goal.append(multitask_rollout(
                env,
                policy,
                max_path_length=args.H,
                render=not args.hide,
                observation_key='observation',
                desired_goal_key='desired_goal',
            ))
            # if hasattr(env, "log_diagnostics"):
            #     env.log_diagnostics(paths_per_goal)
            # if hasattr(env, "get_diagnostics"):
            #     for k, v in env.get_diagnostics(paths_per_goal).items():
            #         logger.record_tabular(k, v)
            # logger.dump_tabular()

            goals.append(dict(ee=set_of_goals[goal_id][0], obj=set_of_goals[goal_id][1]))
            puck_distances.append(paths_per_goal[i]['env_infos'][-1]['puck_distance'])
            hand_distances.append(paths_per_goal[i]['env_infos'][-1]['hand_distance'])
        paths.append(paths_per_goal)

    # ===================== POST-PROCESSING SCOPE =====================
    puck_distances = np.array(puck_distances)
    hand_distances = np.array(hand_distances)

    print('[INFO] Saving results...')
    np.savez_compressed(res_file,
                        puck_distance=puck_distances,
                        hand_distance=hand_distances,
                        goals=goals)
    np.savez_compressed(paths_file, episode=np.array(paths))
    print("Hand distance: mean=%.4f, std=%.4f" % (hand_distances.mean(), hand_distances.std()))
    print("Obj distance: mean=%.4f, std=%.4f" % (puck_distances.mean(), puck_distances.std()))
    print('[INFO] Save done.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=50, help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10, help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str, help='env mode')
    parser.add_argument('--gpu', action='store_false')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--hide', action='store_true')

    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--n_test', type=int, default=1)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--random', action='store_true')
    args_user = parser.parse_args()

    simulate_policy(args_user)
