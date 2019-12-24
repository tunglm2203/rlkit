import numpy as np
import matplotlib.pyplot as plt
import cv2
from multiworld.core.image_env import ImageEnv, unormalize_image

# path = '../data/SawyerPushNIPS-v0_N10000_sawyer_init_camera_zoomed_in_imsize84_random_oracle_split_0.npy'
path = '../data/SawyerPushXYReal-v0_N200_sawyer_init_camera_zoomed_in_imsize84_random_oracle_split_0.npy'

data = np.load(path)

for i in range(1000):
    img = data[i].reshape((3, 84, 84)).transpose()
    # img = img[::-1, :, ::-1]
    cv2.imshow('img', img)
    cv2.waitKey(1)
    input("Press Enter {}".format(i))
