import gym
import multiworld
from multiworld.core.image_env import ImageEnv
import cv2
import numpy as np
from sawyer_control.configs.config import config_dict as config
import time

multiworld.register_real_worl_envs()
config_file = config['tung_config']
n_points = 10   # Resolution workspace for checking, higher is more accurately checking
thresh = 0.05   # Threshold to verify robot reach to correct position or not

# Setting workspace want to check
# min_x, max_x = 0.45, 0.85
# min_y, max_y = -0.25, 0.25
# min_z, max_z = 0.06, 0.07
min_x, max_x = config_file.POSITION_SAFETY_BOX_LOWS[0], config_file.POSITION_SAFETY_BOX_HIGHS[0]
min_y, max_y = config_file.POSITION_SAFETY_BOX_LOWS[1], config_file.POSITION_SAFETY_BOX_HIGHS[1]
min_z, max_z = config_file.POSITION_SAFETY_BOX_LOWS[2], config_file.POSITION_SAFETY_BOX_HIGHS[2]
# min_z, max_z = 0.06, 0.07   # Not used, since consider XY-plane


def check_successful_go_to_pos_xy(env, pos, thres):
    # Don't care about quaternion
    _, _, cur_pos = env.request_observation()
    cur_pos = cur_pos[:3]
    if np.linalg.norm(cur_pos[:2] - pos[:2]) < thres:
        return True
    else:
        return False


def check_successful_go_to_pos_xyz(env, pos, thres):
    # Don't care about quaternion
    _, _, cur_pos = env.request_observation()
    cur_pos = cur_pos[:3]
    if np.linalg.norm(cur_pos[:3] - pos[:3]) < thres:
        return True
    else:
        return False


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    # env = gym.make('SawyerPushXYReal-v0')
    env = gym.make('SawyerReachXYZReal-v0')
    # env.action_mode = 'torque'

    img_env = ImageEnv(env,
                       normalize=True,
                       transpose=False,
                       recompute_reward=False)
    s = img_env.reset()

    input("Press Enter for checking...")
    success = bcolors.OKGREEN + 'Success' + bcolors.ENDC
    failed = bcolors.FAIL + 'Failed' + bcolors.ENDC

    x_coors = np.linspace(min_x, max_x, n_points)
    y_coors = np.linspace(min_y, max_y, n_points)
    z_coors = np.linspace(min_z, max_z, n_points)

    # TUNG:  ======== Checking move to specific point ========
    # pos = np.array([x_coors[-1], y_coors[0], .06])
    # angles = env.request_ik_angles(pos, env._get_joint_angles())
    # env.send_angle_action(angles, pos)
    # exit()
    # TUNG:  ======== Checking move to specific point ========

    for xi in range(len(x_coors)):
        if xi % 2 == 0:
            flip = False
        else:
            flip = True

        for yi in range(len(y_coors)):
            for zi in range(len(z_coors)):
                # Coordinate want to move to
                # if not flip:
                #     pos = np.array([x_coors[xi], y_coors[yi], .06])
                # else:
                #     pos = np.array([x_coors[xi], y_coors[len(y_coors) - yi - 1], .06])
                if not flip:
                    pos = np.array([x_coors[xi], y_coors[yi], z_coors[len(z_coors) - zi - 1]])
                else:
                    pos = np.array([x_coors[xi], y_coors[len(y_coors) - yi - 1], z_coors[len(z_coors) - zi - 1]])
                # Compute required angles by IK service
                angles = env.request_ik_angles(pos, env._get_joint_angles())
                # Move to angles
                env.send_angle_action(angles, pos)
                time.sleep(2)
                # if check_successful_go_to_pos_xy(env, pos, thresh):
                #     print("Moving to (x, y) = (%.4f, %.4f): %s" % (pos[0], pos[1], success))
                # else:
                #     print("Moving to (x, y) = (%.4f, %.4f): %s" % (pos[0], pos[1], failed))
                if check_successful_go_to_pos_xyz(env, pos, thresh):
                    print("Moving to (x, y) = (%.4f, %.4f): %s" % (pos[0], pos[1], success))
                else:
                    print("Moving to (x, y) = (%.4f, %.4f): %s" % (pos[0], pos[1], failed))
    return 0


if __name__ == '__main__':
    main()
