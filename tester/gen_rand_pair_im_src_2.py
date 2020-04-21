# This script run for both multiworld and sawyer_control to test 2 environments
import os
import cv2
import numpy as np
from tqdm import tqdm
import time

import gym
import multiworld
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in_aim_v0
from multiworld.core.image_env import ImageEnv
from multiworld.core.image_env import unormalize_image
multiworld.register_all_envs()


def compute_checking_coor(env, _n_points_ee_x, _n_points_ee_y, _n_points_obj_x, _n_points_obj_y):
    min_x, max_x = env.hand_goal_low[0], env.hand_goal_high[0]
    min_y, max_y = env.hand_goal_low[1], env.hand_goal_high[1]

    puck_min_x, puck_max_x = env.puck_goal_low[0], env.puck_goal_high[0]
    puck_min_y, puck_max_y = env.puck_goal_low[1], env.puck_goal_high[1]

    x_coors = np.linspace(min_x, max_x, _n_points_ee_x)
    y_coors = np.linspace(min_y, max_y, _n_points_ee_y)

    puck_x_coors = np.linspace(puck_min_x, puck_max_x, _n_points_obj_x)
    puck_y_coors = np.linspace(puck_min_y, puck_max_y, _n_points_obj_y)

    print('x_coors: ', x_coors)
    print('y_coors: ', y_coors)
    print('puck_x_coors: ', puck_x_coors)
    print('puck_y_coors: ', puck_y_coors)
    return x_coors, y_coors, puck_x_coors, puck_y_coors


def main():
    # ======================== USER SCOPE  ========================
    env_id = 'SawyerPushNIPSEasy-v0'
    # env_id = 'SawyerPushNIPSCustomEasy-v0'
    # env_id = 'SawyerPushNIPS-v0'
    imsize = 48
    n_trajectories = 50
    horizon = 9
    data_folder = 'rand_pair_sim_src.{}'.format(n_trajectories * (horizon + 1))
    # key_img = 'image_desired_goal'
    key_img = 'image_observation'

    # root_path = '/mnt/hdd/tung/workspace/rlkit/tester'
    root_path = '/home/tung/workspace/rlkit/tester'

    data_folder_real = 'rand_pair_sim_tgt.{}'.format(n_trajectories * (horizon + 1))
    data = np.load(os.path.join(root_path, data_folder_real, 'random_trajectories.npz'))
    episodes = data['data']
    _n_trajectories = len(episodes)
    _horizon = len(episodes[0])
    assert _n_trajectories == n_trajectories and _horizon == horizon + 1, "Not same episodes, horizon"
    # =================== GENERATING DATA SCOPE  ==================
    save_path = os.path.join(root_path, data_folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Create environment
    env_sim = gym.make(env_id)
    env_sim = ImageEnv(env_sim,
                       imsize=imsize,
                       normalize=True,
                       transpose=True,
                       init_camera=sawyer_init_camera_zoomed_in_aim_v0)

    for i in tqdm(range(n_trajectories)):
        for t in range(horizon + 1):
            # Roll-out into environment
            env_sim._goal_xyxy = np.zeros(4)
            env_sim._goal_xyxy[:2] = np.array(episodes[i][t]['ee_pos'][:2])   # EE
            env_sim._goal_xyxy[2:] = np.array(episodes[i][t]['obj_pos'][:2])  # OBJ

            env_sim.set_goal_xyxy(env_sim._goal_xyxy)
            env_sim.reset_mocap_welds()
            env_sim._get_obs()
            env_sim.wrapped_env.set_to_goal(env_sim.wrapped_env.get_goal())

            # Get true state (coordinates)
            # TUNG: don't have this step

            # Get image of state
            img = env_sim._get_flat_img()
            g = img

            # Show image of state
            im_show = cv2.resize(g.reshape(3, imsize, imsize).transpose()[:, :, ::-1], (128, 128))
            cv2.imshow('observation', im_show)
            cv2.waitKey(1)

            # Save image of state
            filename = os.path.join(save_path, 'ep_{}_s_{}'.format(i, t))
            cv2.imwrite(filename + '.png', unormalize_image(im_show))
            np.savez_compressed(filename, im=img)


if __name__ == '__main__':
    main()
