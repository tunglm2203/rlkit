import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout_sim2real
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv
import torch

import gym
import multiworld
from multiworld.core.image_env import ImageEnv
multiworld.register_all_envs()


def wrap_real_env_manually(data_ckpt=None, env_name='SawyerPushXYReal-v0'):
    """
    Return VAE env wrapping real_world env
    :param data_ckpt:
    :param env_name:
    :return:
    """
    env = gym.make(env_name)
    image_env = ImageEnv(env,
                         imsize=data_ckpt['evaluation/env'].imsize,
                         normalize=True,
                         transpose=True)
    # TUNG: ====== Hard code: clone from configuration of data_ckpt['evaluation/env']
    render = False
    reward_params = dict(
        type='latent_distance',
    )
    # ======

    vae_env = VAEWrappedEnv(
        image_env,
        data_ckpt['vae'],
        imsize=image_env.imsize,
        decode_goals=render,
        render_goals=render,
        render_rollouts=render,
        reward_params=reward_params,
    )
    env = vae_env
    return env


def simulate_policy_on_real(args):
    adapted_vae_ckpt = args.vae
    if args.gpu:
        data = torch.load(open(args.file, "rb"))
    else:
        data = torch.load(open(args.file, "rb"), map_location='cpu')
    env = data['evaluation/env']
    policy = data['evaluation/policy']

    print("Policy and environment loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
    if isinstance(env, VAEWrappedEnv) and hasattr(env, 'mode'):
        env.mode(args.mode)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()

    # TUNG: Using a goal sampled from environment
    env._goal_sampling_mode = 'reset_of_env'

    # Re-initialize REAL env wrapped by learned VAE (from SIM)
    env_manual = wrap_real_env_manually(data_ckpt=data, env_name='SawyerPushXYReal-v0')
    if args.gpu:
        ptu.set_gpu_mode(True)
    if isinstance(env_manual, VAEWrappedEnv) and hasattr(env_manual, 'mode'):
        env_manual.mode(args.mode)
    if args.enable_render or hasattr(env_manual, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env_manual.enable_render()

    # Disable re-compute reward since REAL env doesn't provide it
    env_manual.wrapped_env.recompute_reward = False
    env_manual.wrapped_env.wrapped_env.use_gazebo_auto = True

    # Load adapted VAE
    if adapted_vae_ckpt is not None:
        adapted_vae = torch.load(open(os.path.join(adapted_vae_ckpt, 'vae_ckpt.pth'), "rb"))
        env_manual.vae.load_state_dict(adapted_vae['model'])
    else:
        print('[WARNING] Using VAE of source')

    # This piece of code for test learn policy on REAL env
    paths = []
    for _ in range(args.n_test):
        paths.append(multitask_rollout_sim2real(
            env_manual,
            policy,
            max_path_length=args.H,
            render=not args.hide,
            observation_key='observation',
            desired_goal_key='desired_goal',
        ))
        if hasattr(env_manual, "log_diagnostics"):
            env_manual.log_diagnostics(paths)
        if hasattr(env_manual, "get_diagnostics"):
            for k, v in env_manual.get_diagnostics(paths).items():
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
        filename = os.path.join('debug', 'test_policy.npz')
    np.savez_compressed(filename,
                        puck_distance=puck_distance,
                        hand_distance=hand_distance)


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='path to the snapshot file')
parser.add_argument('--vae', type=str, default=None)
parser.add_argument('--H', type=int, default=50, help='Max length of rollout')
parser.add_argument('--speedup', type=float, default=10, help='Speedup')
parser.add_argument('--mode', default='video_env', type=str, help='env mode')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--enable_render', action='store_true')
parser.add_argument('--hide', action='store_false')
parser.add_argument('--n_test', type=int, default=100)
parser.add_argument('--exp', type=str, default=None)
parser.add_argument('--random', action='store_true')
args_user = parser.parse_args()


if __name__ == "__main__":
    """
    Need to provide 2 parameters in list below:
        file: Path to trained policy (e.g.: /path/to/params.pkl)
        vae: Path to trained adapted VAE (e.g.: /path/to/vae_file, contains vae_ckpt.pth) 
    """
    simulate_policy_on_real(args_user)
