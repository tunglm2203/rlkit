# This script run for both multiworld and sawyer_control to test 2 environments
import gym
import time
import cv2
import os
import numpy as np
from tqdm import tqdm

from multiworld.core.image_env import ImageEnv
from multiworld.core.image_env import unormalize_image
import multiworld
multiworld.register_all_envs()
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in_aim_v1


def check_successful_go_to_pos_xy(env, pos, thres):
    # Don't care about quaternion
    _, _, cur_pos = env.request_observation()
    cur_pos = cur_pos[:3]
    if np.linalg.norm(cur_pos[:2] - pos[:2]) < thres:
        return True
    else:
        return False


def set_obj_location(obj_pos, env):
    args = dict(x=obj_pos[0],
                y=obj_pos[1],
                z=obj_pos[2])
    msg = dict(func='set_object_los', args=args)
    env.client.sending(msg, sleep_before=env.config.SLEEP_BEFORE_SENDING_CMD_SOCKET,
                       sleep_after=env.config.SLEEP_BETWEEN_2_CMDS)
    env.client.sending(msg, sleep_before=0,
                       sleep_after=env.config.SLEEP_AFTER_SENDING_CMD_SOCKET)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def compute_checking_coor(env, _n_points_ee_x, _n_points_ee_y, _n_points_obj_x, _n_points_obj_y):
    min_x_r, max_x_r = env.goal_space.low[2], env.goal_space.high[2]
    min_y_r, max_y_r = env.goal_space.low[3], env.goal_space.high[3]

    puck_min_x_r, puck_max_x_r = env.goal_space.low[0], env.goal_space.high[0]
    puck_min_y_r, puck_max_y_r = env.goal_space.low[1], env.goal_space.high[1]

    x_coors = np.linspace(min_x_r, max_x_r, _n_points_ee_x, endpoint=True)
    y_coors = np.linspace(min_y_r, max_y_r, _n_points_ee_y, endpoint=True)

    puck_x_coors = np.linspace(puck_min_x_r, puck_max_x_r, _n_points_obj_x)
    puck_y_coors = np.linspace(puck_min_y_r, puck_max_y_r, _n_points_obj_y)

    print('x_coors: ', x_coors)
    print('y_coors: ', y_coors)
    print('puck_x_coors: ', puck_x_coors)
    print('puck_y_coors: ', puck_y_coors)
    return x_coors, y_coors, puck_x_coors, puck_y_coors


def main():
    # ======================== USER SCOPE  ========================
    # env_id = 'SawyerPushXYReal-v0'
    env_id = 'SawyerPushNIPSCustomEasy-v0'
    # env_id = 'SawyerPushXYRealMedium-v0'
    imsize = 48
    n_trajectories = 1000
    horizon = 0     # 19
    data_folder = 'randlarger_pair_sim_tgt.{}'.format(n_trajectories * (horizon + 1))
    key_img = 'image_desired_goal'
    key_state = 'state_desired_goal'
    # key_img = 'image_observation'
    # key_state = 'state_observation'

    root_path = '/home/tung/workspace/rlkit/tester'

    # =================== GENERATING DATA SCOPE  ==================
    save_path = os.path.join(root_path, data_folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Create environment
    env_sim = gym.make(env_id, random_init=False,
                       hand_goal_low=(-0.2, 0.5),
                       hand_goal_high=(0.2, 0.7),
                       puck_goal_low=(-0.2, 0.5),
                       puck_goal_high=(0.2, 0.7),
                       )
    env_sim = ImageEnv(env_sim,
                       imsize=imsize,
                       normalize=True,
                       transpose=True,
                       init_camera=sawyer_init_camera_zoomed_in_aim_v1)

    episodes = []
    re_generate_start_epoch = 0  # TUNG: Remember to set this param
    for i in tqdm(range(re_generate_start_epoch, n_trajectories)):
        t, episode = 0, []
        while True:
            # Roll-out into environment
            if t == 0:
                o = env_sim.reset()
            else:
                o, _, _, _ = env_sim.step(env_sim.action_space.sample())

            # Get true state (coordinates)
            obs = dict(ee_pos=o[key_state][:2],
                       obj_pos=o[key_state][2:])
            episode.append(obs)

            # Get image of state
            img = o[key_img]
            g = img

            # Show image of state
            im_show = cv2.resize(g.reshape(3, imsize, imsize).transpose()[:, :, ::-1], (128, 128))
            cv2.imshow('observation', im_show)
            cv2.waitKey(1)

            # Save image of state
            filename = os.path.join(save_path, 'ep_{}_s_{}'.format(i, t))
            cv2.imwrite(filename + '.png', unormalize_image(im_show))
            np.savez_compressed(filename, im=img)

            if t == horizon:
                break
            else:
                t += 1
        # Save true state of each episode
        episodes.append(np.array(episode))

    # Save episode's data
    save_traj_path = os.path.join(save_path, 'random_trajectories.npz')
    np.savez_compressed(save_traj_path, data=episodes)


if __name__ == '__main__':
    main()
