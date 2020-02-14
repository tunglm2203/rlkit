import gym
import multiworld

multiworld.register_real_worl_envs()
from multiworld.core.image_env import ImageEnv
import cv2
import numpy as np
from sawyer_control.configs.config import config_dict as config
import time


def main():
    im_size = 48

    env = gym.make('SawyerPushXYReal-v0')
    # env = gym.make('SawyerReachXYZReal-v0')

    # env.fix_goal = True
    # env.fixed_goal = np.array([.50, -0.2, .45, .0])    # (obj_x, obj_y, ee_x, ee_y)
    env.reset()

    img_env = ImageEnv(env,
                       imsize=im_size,
                       normalize=True,
                       transpose=True,
                       recompute_reward=False)
    img_env.wrapped_env.random_init = True

    for i in range(500):
        if (i + 1) % 100 == 0:
            print('Resetting...')
            s = img_env.reset()
        else:
            s, _, _, _ = img_env.step(img_env.action_space.sample())

        im = s['image_observation'].reshape((3, im_size, im_size)).transpose()
        im = im[::-1, :, ::-1]
        cv2.imshow("CV Image", im)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()