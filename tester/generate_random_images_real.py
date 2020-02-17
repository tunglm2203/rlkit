# This script run for both multiworld and sawyer_control to test 2 environments
import gym
import numpy as np
import cv2
import time
from tqdm import tqdm

import multiworld
from multiworld.core.image_env import ImageEnv
from multiworld.core.image_env import unormalize_image
multiworld.register_all_envs()


# USER constant scope
n_samples = 1000     # Resolution workspace for checking, higher is more accurately checking
imsize = 48
n_samples_to_reset = 1


# TUNG: Real
env_real = gym.make('SawyerPushXYReal-v0')
env_real = ImageEnv(env_real,
                    imsize=imsize,
                    normalize=True,
                    transpose=True)
env_real.recompute_reward = False
env_real.wrapped_env.random_init = True     # Random object's position each time reset
for i in tqdm(range(n_samples)):
    if i % n_samples_to_reset == 0:
        o = env_real.reset()
        time.sleep(2)   # Waiting to place object
    else:
        o, _, _, _ = env_real.step(env_real.action_space.sample())
    # img = o['image_observation']
    img = o['image_desired_goal']
    g = img

    im_show = cv2.resize(g.reshape(3, imsize, imsize).transpose()[:, :, ::-1], (128, 128))
    cv2.imshow('observation', im_show)
    cv2.waitKey(1)

    filename = '/home/tung/workspace/rlkit/tester/random_images_real/{}'.format(i)
    cv2.imwrite(filename + '.png', unormalize_image(im_show))
    np.savez_compressed(filename + '.npy', im=img)

