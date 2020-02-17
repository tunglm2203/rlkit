# This script run for both multiworld and sawyer_control to test 2 environments
import gym
import multiworld
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in_aim_v0
from multiworld.core.image_env import ImageEnv
import numpy as np
from multiworld.core.image_env import unormalize_image
import cv2
multiworld.register_all_envs()
import time


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
    # USER constant scope
    n_points_ee_x = 5     # Resolution workspace for checking, higher is more accurately checking
    n_points_ee_y = 5     # Resolution workspace for checking, higher is more accurately checking
    n_points_obj_x = 5    # Resolution workspace for checking, higher is more accurately checking
    n_points_obj_y = 5    # Resolution workspace for checking, higher is more accurately checking
    thresh = 0.05       # Threshold to verify robot reach to correct position or not
    imsize = 48
    z_coor = 0.11

    success = bcolors.OKGREEN + 'Success' + bcolors.ENDC
    failed = bcolors.FAIL + 'Failed' + bcolors.ENDC

    # TUNG: Real
    env_real = gym.make('SawyerPushXYReal-v0')
    env_real = ImageEnv(env_real,
                        imsize=imsize,
                        normalize=True,
                        transpose=True)

    x_coors_r, y_coors_r, puck_x_coors_r, puck_y_coors_r = compute_checking_coor(env_real,
                                                                                 n_points_ee_x,
                                                                                 n_points_ee_y,
                                                                                 n_points_obj_x,
                                                                                 n_points_obj_y)
    pxis = []
    pyis = []
    xis = []
    yis = []
    cnt = 0

    for pxi in range(len(puck_x_coors_r)):
        for pyi in range(len(puck_y_coors_r)):
            for xi in range(len(x_coors_r)):
                # if xi == pxi or xi == pxi - 1 or xi == pxi + 1:
                #     print('Avoid puck pxi={}, xi={}'.format(pxi, xi))
                #     continue
                if xi % 2 == 0:
                    flip = False
                else:
                    flip = True
                for yi in range(len(y_coors_r)):
                    print('[INFO] pxi={}, pyi={}, xi={}, yi={}'.format(pxi, pyi, xi, yi))
                    if not flip:
                        ee_pos = np.array([x_coors_r[xi], y_coors_r[yi], z_coor])
                    else:
                        ee_pos = np.array([x_coors_r[xi], y_coors_r[len(y_coors_r) - yi - 1], z_coor])

                    angles = env_real.request_ik_angles(ee_pos, env_real._get_joint_angles())
                    env_real.send_angle_action(angles, ee_pos)
                    time.sleep(3)
                    if check_successful_go_to_pos_xy(env_real, ee_pos, thresh):
                        print("Moving to (x, y) = (%.4f, %.4f): %s" % (ee_pos[0], ee_pos[1], success))
                    else:
                        print("Moving to (x, y) = (%.4f, %.4f): %s" % (ee_pos[0], ee_pos[1], failed))

                    obj_pos = np.array([puck_x_coors_r[pxi], puck_y_coors_r[pyi],
                                        env_real.pos_object_reset_position[2]])
                    # if pxi == pxis[cnt] and pyi == pyis[cnt] and xi == xis[cnt] and yi == yis[cnt]:
                    set_obj_location(obj_pos, env_real)
                    time.sleep(2)   # Sleep to place object

                    img = env_real._get_flat_img()
                    g = img

                    im_show = cv2.resize(g.reshape(3, imsize, imsize).transpose()[:, :, ::-1], (128, 128))
                    cv2.imshow('goal', im_show)
                    cv2.waitKey(1)
                    filename = '/home/tung/workspace/rlkit/tester/images_real/im_{}_{}_{}_{}'.\
                        format(pxi, pyi, xi, yi)
                    # if pxi == pxis[cnt] and pyi == pyis[cnt] and xi == xis[cnt] and yi == yis[cnt]:
                    #     print('=========Saving=========: {}, {}, {}, {}'.format(pxi, pyi, xi, yi))
                    #     cnt += 1
                    #     if cnt == len(pxis):
                    #         exit()
                    cv2.imwrite(filename + '.png', unormalize_image(im_show))
                    np.savez_compressed(filename + '.npy', im=img)
            print('Finish one location of object')
            time.sleep(2)

    # print('Position action scale: ', env1.position_action_scale)
    # print('Obs space (low): ', env1.config.POSITION_SAFETY_BOX_LOWS)
    # print('Obs space (high): ', env1.config.POSITION_SAFETY_BOX_HIGHS)
    # print('Goal space (low): ', env1.observation_space['desired_goal'].low)
    # print('Goal space (high): ', env1.observation_space['desired_goal'].high)
    # print('Action space (low): ', env1.action_space.low)
    # print('Action space (high): ', env1.action_space.high)


if __name__ == '__main__':
    main()
