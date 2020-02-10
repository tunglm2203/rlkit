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


# def check_successful_go_to_pos_xy(env, pos, thres):
#     # Don't care about quaternion
#     _, _, cur_pos = env.request_observation()
#     cur_pos = cur_pos[:3]
#     if np.linalg.norm(cur_pos[:2] - pos[:2]) < thres:
#         return True
#     else:
#         return False
#
#
# def set_obj_location(obj_pos, env):
#     args = dict(x=obj_pos[0] + 0.05,
#                 y=obj_pos[1] - 0.05,
#                 z=obj_pos[2])
#     msg = dict(func='set_object_los', args=args)
#     env.client.sending(msg, sleep_before=env.config.SLEEP_BEFORE_SENDING_CMD_SOCKET,
#                        sleep_after=env.config.SLEEP_BETWEEN_2_CMDS)
#     env.client.sending(msg, sleep_before=0,
#                        sleep_after=env.config.SLEEP_AFTER_SENDING_CMD_SOCKET)
#
#
# class bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'


# USER constant scope
n_points_ee = 5     # Resolution workspace for checking, higher is more accurately checking
n_points_obj = 5    # Resolution workspace for checking, higher is more accurately checking
thresh = 0.05       # Threshold to verify robot reach to correct position or not
imsize = 48

# success = bcolors.OKGREEN + 'Success' + bcolors.ENDC
# failed = bcolors.FAIL + 'Failed' + bcolors.ENDC

# TUNG: Simulation
env_sim = gym.make('SawyerPushNIPSEasy-v0')
env_sim = ImageEnv(env_sim,
                   imsize=imsize,
                   normalize=True,
                   transpose=True,
                   init_camera=sawyer_init_camera_zoomed_in_aim_v0)
env_sim.reset()

min_x, max_x = env_sim.hand_goal_low[0], env_sim.hand_goal_high[0]
min_y, max_y = env_sim.hand_goal_low[1], env_sim.hand_goal_high[1]

puck_min_x, puck_max_x = env_sim.puck_goal_low[0], env_sim.puck_goal_high[0]
puck_min_y, puck_max_y = env_sim.puck_goal_low[1], env_sim.puck_goal_high[1]

x_coors = np.linspace(min_x, max_x, n_points_ee)
y_coors = np.linspace(min_y, max_y, n_points_ee)

puck_x_coors = np.linspace(puck_min_x, puck_max_x, n_points_obj)
puck_y_coors = np.linspace(puck_min_y, puck_max_y, n_points_obj)

print('x_coors: ', x_coors)
print('y_coors: ', y_coors)
print('puck_x_coors: ', puck_x_coors)
print('puck_y_coors: ', puck_y_coors)

for pxi in range(len(puck_x_coors)):
    for pyi in range(len(puck_y_coors)):
        for xi in range(len(x_coors)):
            if xi == pxi or xi == pxi - 1 or xi == pxi + 1:
                print('Avoid puck pxi={}, xi={}'.format(pxi, xi))
                continue
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

# TUNG: Real
# env_real = gym.make('SawyerPushXYReal-v0')
# env_real = ImageEnv(env_real,
#                     normalize=True,
#                     transpose=True)
# env_real.reset()

# min_x_r, max_x_r = env_real.goal_space.low[2], env_real.goal_space.high[2]
# min_y_r, max_y_r = env_real.goal_space.low[3], env_real.goal_space.high[3]
#
# puck_min_x_r, puck_max_x_r = env_real.goal_space.low[0], env_real.goal_space.high[0]
# puck_min_y_r, puck_max_y_r = env_real.goal_space.low[1], env_real.goal_space.high[1]
#
# x_coors_r = np.linspace(min_x_r, max_x_r, n_points_ee, endpoint=True)
# y_coors_r = np.linspace(min_y_r, max_y_r, n_points_ee, endpoint=True)
#
# puck_x_coors_r = np.linspace(puck_min_x_r, puck_max_x_r, n_points_obj)
# puck_y_coors_r = np.linspace(puck_min_y_r, puck_max_y_r, n_points_obj)
#
# print('x_coors: ', x_coors_r)
# print('y_coors: ', y_coors_r)
# print('puck_x_coors: ', puck_x_coors_r)
# print('puck_y_coors: ', puck_y_coors_r)
#
# for pxi in range(len(puck_x_coors)):
#     for pyi in range(len(puck_y_coors)):
#         obj_pos = np.array([puck_x_coors[pxi], puck_y_coors[pyi],
#                             env_real.pos_object_reset_position[2]])
#         set_obj_location(obj_pos, env_real)
#         for xi in range(len(x_coors)):
#             if xi % 2 == 0:
#                 flip = False
#             else:
#                 flip = True
#             for yi in range(len(y_coors)):
#                 if not flip:
#                     ee_pos = np.array([x_coors[xi], y_coors[yi], .06])
#                 else:
#                     ee_pos = np.array([x_coors[xi], y_coors[len(y_coors) - yi - 1], .06])
#
#                 angles = env_real.request_ik_angles(ee_pos, env_real._get_joint_angles())
#                 env_real.send_angle_action(angles, ee_pos)
#                 time.sleep(2)
#                 if check_successful_go_to_pos_xy(env_real, ee_pos, thresh):
#                     print("Moving to (x, y) = (%.4f, %.4f): %s" % (ee_pos[0], ee_pos[1], success))
#                 else:
#                     print("Moving to (x, y) = (%.4f, %.4f): %s" % (ee_pos[0], ee_pos[1], failed))

# print('Position action scale: ', env1.position_action_scale)
# print('Obs space (low): ', env1.config.POSITION_SAFETY_BOX_LOWS)
# print('Obs space (high): ', env1.config.POSITION_SAFETY_BOX_HIGHS)
# print('Goal space (low): ', env1.observation_space['desired_goal'].low)
# print('Goal space (high): ', env1.observation_space['desired_goal'].high)
# print('Action space (low): ', env1.action_space.low)
# print('Action space (high): ', env1.action_space.high)
