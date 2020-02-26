# This script run for both multiworld and sawyer_control to test 2 environments
import numpy as np
import gym
import cv2
import os
import time
from tqdm import tqdm

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
    # USER constant scope
    imsize = 48
    n_trajectories = 50
    horizon = 19
    root_path = '/home/tung/workspace/rlkit/tester/random_pair_sim'

    # TUNG: Simulation
    env_sim = gym.make('SawyerPushNIPSEasy-v0')
    env_sim = ImageEnv(env_sim,
                       imsize=imsize,
                       normalize=True,
                       transpose=True,
                       init_camera=sawyer_init_camera_zoomed_in_aim_v0)
    episodes = []
    for i in tqdm(range(n_trajectories)):
        t = 0
        episode = []
        while True:
            if t == 0:
                o = env_sim.reset()
            else:
                o, _, _, _ = env_sim.step(env_sim.action_space.sample())

            episode.append(o['state_observation'])

            img = o['image_observation']
            g = img

            im_show = cv2.resize(g.reshape(3, imsize, imsize).transpose()[:, :, ::-1], (128, 128))
            cv2.imshow('observation', im_show)
            cv2.waitKey(1)
            filename = os.path.join(root_path, 'ep_{}_s_{}'.format(i, t))
            cv2.imwrite(filename + '.png', unormalize_image(im_show))
            np.savez_compressed(filename + '.npy', im=img)
            if t == horizon:
                break
            else:
                t += 1
        episodes.append(np.array(episode))

    save_traj_path = os.path.join(root_path, 'random_trajectories.npz')
    np.savez_compressed(save_traj_path, data=episodes)


if __name__ == '__main__':
    main()
