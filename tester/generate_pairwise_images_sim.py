# This script run for both multiworld and sawyer_control to test 2 environments
import gym
import multiworld
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in_aim_v0
from multiworld.core.image_env import ImageEnv
from multiworld.core.image_env import unormalize_image
import numpy as np
import cv2
multiworld.register_all_envs()
import time


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
    n_points_ee_x = 5  # Resolution workspace for checking, higher is more accurately checking
    n_points_ee_y = 5  # Resolution workspace for checking, higher is more accurately checking
    n_points_obj_x = 5  # Resolution workspace for checking, higher is more accurately checking
    n_points_obj_y = 5  # Resolution workspace for checking, higher is more accurately checking
    imsize = 48

    # TUNG: Simulation
    env_sim = gym.make('SawyerPushNIPSEasy-v0')
    env_sim = ImageEnv(env_sim,
                       imsize=imsize,
                       normalize=True,
                       transpose=True,
                       init_camera=sawyer_init_camera_zoomed_in_aim_v0)
    env_sim.reset()

    x_coors, y_coors, puck_x_coors, puck_y_coors = compute_checking_coor(env_sim,
                                                                         n_points_ee_x,
                                                                         n_points_ee_y,
                                                                         n_points_obj_x,
                                                                         n_points_obj_y)

    for pxi in range(len(puck_x_coors)):
        for pyi in range(len(puck_y_coors)):
            for xi in range(len(x_coors)):
                # if xi == pxi or xi == pxi - 1 or xi == pxi + 1:
                #     print('Avoid puck pxi={}, xi={}'.format(pxi, xi))
                #     continue
                if xi % 2 == 0:
                    flip = False
                else:
                    flip = True
                for yi in range(len(y_coors)):
                    env_sim._goal_xyxy = np.zeros(4)
                    if not flip:
                        env_sim._goal_xyxy[:2] = np.array([x_coors[xi], y_coors[yi]])  # EE
                    else:
                        env_sim._goal_xyxy[:2] = np.array([x_coors[xi], y_coors[len(y_coors) - yi - 1]])  # EE

                    env_sim._goal_xyxy[2:] = np.array([puck_x_coors[pxi] + 0.05,
                                                       puck_y_coors[pyi] - 0.05])   # OBJ
                    env_sim.set_goal_xyxy(env_sim._goal_xyxy)
                    # env_sim.set_puck_xy(env_sim.sample_puck_xy())
                    env_sim.reset_mocap_welds()
                    env_sim._get_obs()

                    env_state = env_sim.wrapped_env.get_env_state()
                    env_sim.wrapped_env.set_to_goal(env_sim.wrapped_env.get_goal())
                    img = env_sim._get_flat_img()
                    env_sim.wrapped_env.set_env_state(env_state)
                    g = img

                    im_show = cv2.resize(g.reshape(3, imsize, imsize).transpose()[:, :, ::-1], (128, 128))
                    cv2.imshow('goal', im_show)
                    cv2.waitKey(1)
                    filename = '/home/tung/workspace/rlkit/tester/images_sim/im_{}_{}_{}_{}'. \
                        format(pxi, pyi, xi, yi)
                    cv2.imwrite(filename + '.png', unormalize_image(im_show))
                    np.savez_compressed(filename + '.npy', im=img)


if __name__ == '__main__':
    main()
