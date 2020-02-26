# This script run for both multiworld and sawyer_control to test 2 environments
import gym
import multiworld
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in_aim_v0
from multiworld.core.image_env import ImageEnv
from multiworld.core.image_env import unormalize_image
import numpy as np
import cv2
multiworld.register_all_envs()
from tqdm import tqdm
import time


def main():
    # USER constant scope
    n_samples = 1000  # Resolution workspace for checking, higher is more accurately checking
    imsize = 48
    n_samples_to_reset = 1

    # TUNG: Simulation
    env_sim = gym.make('SawyerPushNIPSEasy-v0')
    env_sim = ImageEnv(env_sim,
                       imsize=imsize,
                       normalize=True,
                       transpose=True,
                       init_camera=sawyer_init_camera_zoomed_in_aim_v0)
    env_sim.reset()

    for i in tqdm(range(n_samples)):
        if i % n_samples_to_reset == 0:
            o = env_sim.reset()
        else:
            o, _, _, _ = env_sim.step(env_sim.action_space.sample())
        # img = o['image_observation']
        img = o['image_desired_goal']
        g = img

        im_show = cv2.resize(g.reshape(3, imsize, imsize).transpose()[:, :, ::-1], (128, 128))
        cv2.imshow('observation', im_show)
        cv2.waitKey(1)

        filename = '/home/tung/workspace/rlkit/tester/random_images_sim/{}'.format(i)
        cv2.imwrite(filename + '.png', unormalize_image(im_show))
        np.savez_compressed(filename + '.npy', im=img)


if __name__ == '__main__':
    main()
