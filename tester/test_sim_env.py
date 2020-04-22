import multiworld
import gym
import cv2

multiworld.register_mujoco_envs()

from rlkit.sim2real.sim2real_utils import set_env_state_sim2sim
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in_aim_v0
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in_aim_v1

from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera


def test_sim():
    # env = gym.make('SawyerPushNIPS-v0')
    # env = gym.make('SawyerPushNIPSEasy-v0')
    env = gym.make('SawyerPushNIPSCustomEasy-v0')
    # env = gym.make('SawyerPushNIPSHarder-v0')
    # env = gym.make('SawyerPickupEnvYZEasy-v0')

    use_image_env = True
    im_size = 512
    if use_image_env:
        img_env = ImageEnv(env,
                           imsize=im_size,
                           normalize=True,
                           transpose=True,
                           # init_camera=sawyer_init_camera_zoomed_in_aim_v0,
                           init_camera=sawyer_init_camera_zoomed_in_aim_v1,
                           # init_camera=sawyer_pick_and_place_camera,
                           )
    else:
        img_env = env

    n_steps_to_reset = 10000
    for i in range(10000):
        if i % n_steps_to_reset == 0:
            s = img_env.reset()
            # input("Press Enter...")
        else:
            s, _, _, _ = img_env.step(img_env.action_space.sample())

        image_key = 'image_observation'
        # image_key = 'image_desired_goal'
        if use_image_env:
            im = s[image_key].reshape((3, im_size, im_size)).transpose()
            im = im[::-1, :, ::-1]
            cv2.imshow("CV Image", im)
            cv2.waitKey(1)
        else:
            img_env.render()


def test_set_obj():
    import numpy as np
    import time
    im_size = 256

    env1 = gym.make('SawyerPushNIPSEasy-v0')
    img_env1 = ImageEnv(env1,
                        imsize=im_size,
                        normalize=True,
                        transpose=True,
                        init_camera=sawyer_init_camera_zoomed_in_aim_v0,
                        # init_camera=sawyer_init_camera_zoomed_in_aim_v1,
                        # init_camera=sawyer_pick_and_place_camera,
                        )

    env2 = gym.make('SawyerPushNIPSEasy-v0')
    img_env2 = ImageEnv(env2,
                        imsize=im_size,
                        normalize=True,
                        transpose=True,
                        init_camera=sawyer_init_camera_zoomed_in_aim_v0,
                        # init_camera=sawyer_init_camera_zoomed_in_aim_v1,
                        # init_camera=sawyer_pick_and_place_camera,
                        )
    s1 = img_env1.reset()
    s2 = img_env2.reset()

    # image_key = 'image_observation'
    image_key = 'image_desired_goal'
    for i in range(10000):
        s1, _, _, _ = img_env1.step(img_env1.action_space.sample())
        set_env_state_sim2sim(src=img_env1, target=img_env2, set_goal=False)

        # s1 = img_env1.reset()
        # set_env_state_sim2sim(src=img_env1, target=img_env2, set_goal=True)

        # Get image on Mujoco
        img = img_env2._get_flat_img()

        im = s1[image_key].reshape((3, im_size, im_size)).transpose()
        im = im[::-1, :, ::-1]
        cv2.imshow("Env1", im)
        cv2.waitKey(1)

        im = img.reshape((3, im_size, im_size)).transpose()
        im = im[::-1, :, ::-1]
        cv2.imshow("Env2", im)
        cv2.waitKey(1)

        img_sub = s1[image_key] - img
        im = img_sub.reshape((3, im_size, im_size)).transpose()
        im = im[::-1, :, ::-1]
        cv2.imshow("Env1-Env2", im)
        cv2.waitKey(1)

        time.sleep(0.1)
        print("DEBUG: ", np.mean(img - s1[image_key]))


if __name__ == '__main__':
    # test_sim()
    test_set_obj()
