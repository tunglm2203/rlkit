import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv
import torch


def simulate_policy(args):
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

    # TUNG:
    env._goal_sampling_mode = 'reset_of_env'

    paths = []
    for _ in range(args.n_test):
        paths.append(multitask_rollout(
            env,
            policy,
            max_path_length=args.H,
            render=not args.hide,
            observation_key='observation',
            desired_goal_key='desired_goal',
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()

    puck_distance, hand_distance = [], []
    for i in range(args.n_test):
        puck_distance.append(paths[i]['env_infos'][-1]['puck_distance'])
        hand_distance.append(paths[i]['env_infos'][-1]['hand_distance'])

    puck_distance = np.array(puck_distance)
    hand_distance = np.array(hand_distance)

    if args.exp is not None:
        filename = os.path.join('debug', 'test_policy_' + args.exp + '.npz')
    else:
        filename = os.path.join('debug', 'test_policy_sim.npz')
    np.savez_compressed(filename,
                        puck_distance=puck_distance,
                        hand_distance=hand_distance)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=50,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_false')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--exp', type=str, default=None)
    args = parser.parse_args()

    simulate_policy(args)
