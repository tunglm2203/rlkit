# This script run for both multiworld and sawyer_control to test 2 environments
import os
import cv2
import numpy as np
from tqdm import tqdm
import time

import gym
import multiworld
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in_aim_v0
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in_aim_v1
from multiworld.core.image_env import ImageEnv
from multiworld.core.image_env import unormalize_image
multiworld.register_all_envs()


def main():
    # ======================== USER SCOPE  ========================
    root_path = '/home/tung/workspace/rlkit/tester/'

    env_id = 'SawyerPushNIPSEasy-v0'
    # env_id = 'SawyerPushNIPSCustomEasy-v0'
    # env_id = 'SawyerPushNIPS-v0'

    n_samples = 500
    imsize = 48
    n_samples_to_reset = 1
    data_folder = 'rand_img_sim.{}'.format(n_samples)
    # data_folder = 'rand_img_sim_tgt.{}'.format(n_samples)
    key_img = 'image_desired_goal'
    # key_img = 'image_observation'

    # =================== GENERATING DATA SCOPE  ==================
    save_path = os.path.join(root_path, data_folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # Create environment
    env_sim = gym.make(env_id)
    env_sim = ImageEnv(env_sim,
                       imsize=imsize,
                       normalize=True,
                       transpose=True,
                       init_camera=sawyer_init_camera_zoomed_in_aim_v0,
                       # init_camera=sawyer_init_camera_zoomed_in_aim_v1,
                       )
    env_sim.reset()

    # Generating data
    for i in tqdm(range(n_samples)):
        if i % n_samples_to_reset == 0:
            o = env_sim.reset()
        else:
            o, _, _, _ = env_sim.step(env_sim.action_space.sample())
        img = o[key_img]
        g = img

        im_show = cv2.resize(g.reshape(3, imsize, imsize).transpose()[:, :, ::-1], (128, 128))
        cv2.imshow('observation', im_show)
        cv2.waitKey(1)

        filename = os.path.join(save_path, '{}'.format(i))
        cv2.imwrite(filename + '.png', unormalize_image(im_show))
        np.savez_compressed(filename, im=img)


if __name__ == '__main__':
    main()
