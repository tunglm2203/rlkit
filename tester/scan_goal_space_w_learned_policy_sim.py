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
    # ================== LOGGER SCOPE ==================
    # Check path of stored directory
    if not os.path.exists(args.result_path):
        print('[ERROR-AIM] The directory to store result is not exist: ', args.result_path)
        return
    if args.exp == '':
        print('[WARNING-AIM] You should set name of experiment.')
    saved_dir = os.path.join(args.result_path, args.exp)
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    with open(os.path.join(saved_dir, 'log_params.json'), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, cls=MyEncoder)

    f = open(os.path.join(saved_dir, 'perfomance.txt'), 'w')

    # ================== LOAD DATA SCOPE ==================
    data = torch.load(open(args.file, "rb"))
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    print("Policy and environment loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
    if isinstance(env, VAEWrappedEnv) and hasattr(env, 'mode'):
        env.mode(args.mode)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()

    # Config VAE Env:
    env._goal_sampling_mode = 'reset_of_env'
    # Set goal to set of fixed goals
    env.wrapped_env.wrapped_env.randomize_goals = False

    # ================== CONFIG GOAL's SPACE SCOPE ==================
    ee_x_range = [0.55, 0.6, 0.65]
    ee_y_range = [-0.1, -0.05, 0.0, 0.05, 0.1]
    obj_x_range = [0.5, 0.55, 0.6, 0.65, 0.7]
    obj_y_range = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]

    ee_coors, ob_coors, puck_infos, hand_infos = [], [], [], []
    n_goals = len(ee_x_range) * len(ee_y_range) * len(obj_x_range) * len(obj_y_range)
    cnt = 0
    for x_ee in tqdm(range(len(ee_x_range))):
        for y_ee in range(len(ee_y_range)):
            for x_obj in range(len(obj_x_range)):
                for y_obj in range(len(obj_y_range)):
                    cnt += 1
                    print('[AIM-INFO] Counting: {}/{}'.format(cnt, n_goals))
                    env.wrapped_env.wrapped_env.fixed_hand_goal = np.array([ee_x_range[x_ee],
                                                                            ee_y_range[y_ee]])
                    env.wrapped_env.wrapped_env.fixed_puck_goal = np.array([obj_x_range[x_obj],
                                                                            obj_y_range[y_obj]])
                    paths, puck_distance, hand_distance = [], [], []
                    for _ in tqdm(range(args.n_test)):
                        paths.append(multitask_rollout(
                            env,
                            policy,
                            max_path_length=args.H,
                            render=not args.hide,
                            observation_key='observation',
                            desired_goal_key='desired_goal',
                        ))

                    for i in range(args.n_test):
                        puck_distance.append(paths[i]['env_infos'][-1]['puck_distance'])
                        hand_distance.append(paths[i]['env_infos'][-1]['hand_distance'])

                    puck_distance = np.array(puck_distance)
                    hand_distance = np.array(hand_distance)

                    ee_coors.append(env.wrapped_env.wrapped_env.fixed_hand_goal)
                    ob_coors.append(env.wrapped_env.wrapped_env.fixed_puck_goal)
                    hand_infos.append((hand_distance.mean(), hand_distance.std()))
                    puck_infos.append((puck_distance.mean(), puck_distance.std()))

                    f.write("hand_goal={}, obj_goal={}\n".format(env.wrapped_env.wrapped_env.fixed_hand_goal, env.wrapped_env.wrapped_env.fixed_puck_goal))
                    f.write("Hand distance: mean=%.4f, std=%.4f\n" % (hand_distance.mean(), hand_distance.std()))
                    f.write("Obj distance: mean=%.4f, std=%.4f\n" % (puck_distance.mean(), puck_distance.std()))
                    f.write("\n")
    f.close()
    np.savez_compressed(os.path.join(saved_dir, 'results.npz'),
                        ee_coors=ee_coors,
                        ob_coors=ob_coors,
                        hand_infos=hand_infos,
                        puck_infos=puck_infos)


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
    parser.add_argument('--n_test', type=int, default=10)
    parser.add_argument('--exp', type=str, default='scan_goal')
    args_user = parser.parse_args()

    simulate_policy(args_user)
