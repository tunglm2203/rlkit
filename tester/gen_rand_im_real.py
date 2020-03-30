# This script run for both multiworld and sawyer_control to test 2 environments
import os
import cv2
import numpy as np
from tqdm import tqdm
import time

import gym
import multiworld
from multiworld.core.image_env import ImageEnv
from multiworld.core.image_env import unormalize_image
multiworld.register_all_envs()


def main():
    # ======================== USER SCOPE  ========================
    env_id = 'SawyerPushXYReal-v0'
    # env_id = 'SawyerPushXYRealMedium-v0'
    n_samples = 10
    imsize = 48
    n_samples_to_reset = 1
    data_folder = 'rand_img_real.{}'.format(n_samples)
    key_img = 'image_desired_goal'
    # key_img = 'image_observation'

    root_path = '/home/tung/workspace/rlkit/tester/'

    # =================== GENERATING DATA SCOPE  ==================
    save_path = os.path.join(root_path, data_folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # Create environment
    env_real = gym.make(env_id, use_gazebo_auto=True)
    env_real = ImageEnv(env_real,
                        imsize=imsize,
                        normalize=True,
                        transpose=True)
    env_real.recompute_reward = False

    # Generating data (reset every loop)
    for i in tqdm(range(n_samples)):
        goal = env_real.wrapped_env.sample_goal()
        env_real.wrapped_env.set_to_goal(goal)
        img = env_real._get_flat_img()
        g = img

        im_show = cv2.resize(g.reshape(3, imsize, imsize).transpose()[:, :, ::-1], (128, 128))
        cv2.imshow('observation', im_show)
        cv2.waitKey(1)

        filename = os.path.join(save_path, '{}'.format(i))
        cv2.imwrite(filename + '.png', unormalize_image(im_show))
        np.savez_compressed(filename, im=img)


if __name__ == '__main__':
    main()
