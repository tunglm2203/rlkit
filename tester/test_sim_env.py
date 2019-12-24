import multiworld
import gym
import cv2

multiworld.register_mujoco_envs()
env = gym.make('SawyerPushNIPS-v0')

from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in

use_image_env = True

if use_image_env:
    img_env = ImageEnv(env,
                       normalize=True,
                       init_camera=sawyer_init_camera_zoomed_in,
                       transpose=False)
else:
    img_env = env

s = img_env.reset()
n_steps_to_reset = 500
for i in range(10000):
    if i % n_steps_to_reset == 0 and i != 0:
        s = img_env.reset()
        input("Press Enter...")
    if use_image_env:
        im = s['image_observation'].reshape((3, 84, 84)).transpose()
        im = im[::-1, :, ::-1]
        cv2.imshow("CV Image", im)
        cv2.waitKey(5)
    else:
        img_env.render()
        cv2.waitKey(5)

    img_env.step(img_env.action_space.sample())



