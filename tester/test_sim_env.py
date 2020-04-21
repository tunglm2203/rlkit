import multiworld
import gym
import cv2

multiworld.register_mujoco_envs()

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


if __name__ == '__main__':
    test_sim()
