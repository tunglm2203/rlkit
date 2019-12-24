import gym
import multiworld

multiworld.register_real_worl_envs()
from multiworld.core.image_env import ImageEnv
import cv2
import numpy as np
from sawyer_control.configs.config import config_dict as config
import time


def main():
    env = gym.make('SawyerPushXYReal-v0')
    # env = gym.make('SawyerReachXYZReal-v0')

    # env.fix_goal = True
    # env.fixed_goal = np.array([.50, -0.2, .45, .0])    # (obj_x, obj_y, ee_x, ee_y)

    img_env = ImageEnv(env,
                       normalize=True,
                       transpose=False,
                       recompute_reward=False)

    s = img_env.reset()

    for i in range(500):
        im = s['image_observation'].reshape((3, 84, 84)).transpose()
        s, r, done, info = img_env.step(img_env.action_space.sample())
        print('Reward: ', r)
        print('Done: ', done)
        print('Info: ', info)
        if (i + 1) % 10 == 0:
            s = img_env.reset()
        cv2.imshow("CV Image", im)
        cv2.waitKey(5)


if __name__ == '__main__':
    main()
