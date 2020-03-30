import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

from rlkit.core import logger
from rlkit.core.logging import MyEncoder
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
    # ===================== LOGGER SCOPE =====================
    # Check path of stored directory
    if not os.path.exists(args.result_path):
        print('[ERROR-AIM] The directory to store result is not exist.')
        return

    if args.exp is None:
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

    # Re-initialize REAL env wrapped by learned VAE (from SIM)
    env_manual = wrap_real_env_manually(data_ckpt=data, env_name='SawyerPushXYReal-v0')
    if args.gpu:
        ptu.set_gpu_mode(True)
    if isinstance(env_manual, VAEWrappedEnv) and hasattr(env_manual, 'mode'):
        env_manual.mode(args.mode)
    if args.enable_render or hasattr(env_manual, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env_manual.enable_render()

    # Load adapted VAE
    if adapted_vae_ckpt is not None:
        if args.gpu:
            adapted_vae = torch.load(open(os.path.join(adapted_vae_ckpt, 'vae_ckpt.pth'), "rb"))
        else:
            adapted_vae = torch.load(open(os.path.join(adapted_vae_ckpt, 'vae_ckpt.pth'), "rb"),
                                     map_location='cpu')
        env_manual.vae.load_state_dict(adapted_vae['model'])
    else:
        print('[WARNING] Using VAE of source')

    # ===================== SETUP ENVIRONMENT SCOPE =====================
    # TUNG: Using a goal sampled from environment
    env_manual._goal_sampling_mode = 'reset_of_env'
    env_manual.wrapped_env.wrapped_env.randomize_goals = False
    # Disable re-compute reward since REAL env doesn't provide it
    env_manual.wrapped_env.recompute_reward = False
    env_manual.wrapped_env.wrapped_env.use_gazebo_auto = True

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
        env_manual.wrapped_env.wrapped_env.fixed_hand_goal = set_of_goals[goal_id][0]
        env_manual.wrapped_env.wrapped_env.fixed_puck_goal = set_of_goals[goal_id][1]

        # Number of test per each goal
        n_test, paths_per_goal = 0, []
        while n_test < args.n_test:
            store = False
            print('[INFO] Goal: {}/{}, test count: {}/{}'.format(goal_id, n_goals,
                                                                 n_test, args.n_test))
            path = multitask_rollout_sim2real(
                env_manual,
                policy,
                max_path_length=args.H,
                render=not args.hide,
                observation_key='observation',
                desired_goal_key='desired_goal',
                adapt=args.no_adapt
            )
            if args.monitor:
                user_input = input("Press 'y' to save this trajectory: ")
                if user_input == 'y' or user_input == 'Y':
                    print('[INFO] Store this trajectory')
                    paths_per_goal.append(path)
                    n_test += 1
                    store = True
                else:
                    print('[INFO] Discard this trajectory')
            else:
                paths_per_goal.append(path)
                n_test += 1
                store = True

            # if hasattr(env_manual, "log_diagnostics"):
            #     env_manual.log_diagnostics(paths_per_goal)
            # if hasattr(env_manual, "get_diagnostics"):
            #     for k, v in env_manual.get_diagnostics(paths_per_goal).items():
            #         logger.record_tabular(k, v)
            # logger.dump_tabular()

            if store:
                goals.append(dict(ee=set_of_goals[goal_id][0], obj=set_of_goals[goal_id][1]))
                puck_distances.append(path['env_infos'][-1]['puck_distance'])
                hand_distances.append(path['env_infos'][-1]['hand_distance'])
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


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='path to the snapshot file')
parser.add_argument('--vae', type=str, default=None)
parser.add_argument('--H', type=int, default=50, help='Max length of rollout')
parser.add_argument('--mode', default='video_env', type=str, help='env mode')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--enable_render', action='store_true')
parser.add_argument('--hide', action='store_false')

parser.add_argument('--monitor', action='store_true')
parser.add_argument('--result_path', type=str, default='results')
parser.add_argument('--n_test', type=int, default=1)
parser.add_argument('--exp', type=str, default='')
parser.add_argument('--random', action='store_true')
parser.add_argument('--no_adapt', action='store_false')
args_user = parser.parse_args()


if __name__ == "__main__":
    """
    Need to provide 2 parameters in list below:
        file: Path to trained policy (e.g.: /path/to/params.pkl)
        vae: Path to trained adapted VAE (e.g.: /path/to/vae_file, contains vae_ckpt.pth) 
    """
    simulate_policy_on_real(args_user)
