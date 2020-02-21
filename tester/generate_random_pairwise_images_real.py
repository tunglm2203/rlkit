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


def check_successful_go_to_pos_xy(env, pos, thres):
    # Don't care about quaternion
    _, _, cur_pos = env.request_observation()
    cur_pos = cur_pos[:3]
    if np.linalg.norm(cur_pos[:2] - pos[:2]) < thres:
        return True
    else:
        return False


def set_obj_location(obj_pos, env):
    # args = dict(x=obj_pos[0] + 0.05,
    #             y=obj_pos[1] - 0.05,
    #             z=obj_pos[2])
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
    data = np.load('/home/tung/workspace/rlkit/tester/random_pair_sim/random_trajectories.npz')
    data = data['data']
    # USER constant scope
    thresh = 0.04       # Threshold to verify robot reach to correct position or not
    imsize = 48
    z_coor = 0.11    # 0.11

    obj_name = 'cylinder'
    n_trajectories = len(data)
    horizon = len(data[0])
    root_path = '/home/tung/workspace/rlkit/tester/random_pair_real'

    success = bcolors.OKGREEN + 'Success' + bcolors.ENDC
    failed = bcolors.FAIL + 'Failed' + bcolors.ENDC

    print('Number of trajectories: ', n_trajectories)
    print('Horizon: ', horizon)

    # TUNG: Real
    env_real = gym.make('SawyerPushXYReal-v0')
    env_real = ImageEnv(env_real,
                        imsize=imsize,
                        normalize=True,
                        transpose=True)
    env_real.wrapped_env.use_gazebo_auto = True

    ee_pos = np.zeros(3)
    obj_pos = np.zeros(3)
    env_real.reset()
    re_generate_start_epoch = 0  # TUNG: Remember to set this param
    for i in tqdm(range(re_generate_start_epoch, n_trajectories)):
        for t in range(horizon):
            ee_pos[:2], ee_pos[2] = data[i][t][:2], z_coor
            obj_pos[:2], obj_pos[2] = data[i][t][2:], env_real.pos_object_reset_position[2]
            angles = env_real.request_ik_angles(ee_pos, env_real._get_joint_angles())
            env_real.send_angle_action(angles, ee_pos)
            if check_successful_go_to_pos_xy(env_real, ee_pos, thresh):
                print("Moving to (x, y) = (%.4f, %.4f): %s" % (ee_pos[0], ee_pos[1], success))
            else:
                print("Moving to (x, y) = (%.4f, %.4f): %s" % (ee_pos[0], ee_pos[1], failed))

            env_real.wrapped_env.set_obj_to_pos_in_gazebo(obj_name, obj_pos)

            img = env_real._get_flat_img()
            g = img

            im_show = cv2.resize(g.reshape(3, imsize, imsize).transpose()[:, :, ::-1], (128, 128))
            cv2.imshow('observation', im_show)
            cv2.waitKey(1)
            filename = os.path.join(root_path, 'ep_{}_s_{}'.format(i, t))
            cv2.imwrite(filename + '.png', unormalize_image(im_show))
            np.savez_compressed(filename + '.npy', im=img)


if __name__ == '__main__':
    main()
