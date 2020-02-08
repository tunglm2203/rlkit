import argparse
import glob
import os
import time
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

import torch

from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv


def load_data(path=''):
    images = []
    if os.path.exists(path):
        filenames = glob.glob(os.path.join(path, '*.npz'))
        filenames.sort()
        N = len(filenames)
        for i in range(N):
            im = np.load(filenames[i])
            images.append(im['im'])
    else:
        print('Cannot find data')
    return np.array(images)


def main(args):
    # Load images
    print('Loading images...')
    path_to_images_r = '/mnt/hdd/tung/workspace/rlkit/tester/images_real'
    path_to_images_s = '/mnt/hdd/tung/workspace/rlkit/tester/images_sim'
    images_r = load_data(path=path_to_images_r)
    images_s = load_data(path=path_to_images_s)
    # Load pre-trained policy
    print("Loading policy and environment...")
    sim_data = torch.load(open(args.file, "rb"))
    env = sim_data['evaluation/env']

    if args.gpu:
        ptu.set_gpu_mode(True)
    if isinstance(env, VAEWrappedEnv) and hasattr(env, 'mode'):
        env.mode(args.mode)

    src_vae = env.vae   # Get learned VAE from env

    if args.vae is not None:
        adapted_vae = torch.load(open(os.path.join(args.vae, 'ckpt.pth'), "rb"))
        tgt_vae = copy.deepcopy(src_vae)
        # TODO: load trained aptated VAE here, continue...
        tgt_vae.load_state_dict(adapted_vae['model'])

    assert images_r.dtype == np.float64
    assert images_s.dtype == np.float64
    imgs_r = ptu.from_numpy(images_r)
    imgs_s = ptu.from_numpy(images_s)
    latents = None
    if len(images_r) > 0:
        if args.vae is not None:
            latent_distribution_params_r = tgt_vae.encode(imgs_r)
        else:
            latent_distribution_params_r = src_vae.encode(imgs_r)

        latent_distribution_params_s = src_vae.encode(imgs_s)
        latents_r = ptu.get_numpy(latent_distribution_params_r[0])
        latents_s = ptu.get_numpy(latent_distribution_params_s[0])
        # Merge sim & real data together to make same relative coordinate
        latents = np.concatenate((latents_s, latents_r), axis=0)

    # T-SNE
    time_start = time.time()
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latents)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    plt.figure()
    ax = plt.axes(projection='3d')
    xs_r = tsne_results[:, 0]
    ys_r = tsne_results[:, 1]
    zs_r = tsne_results[:, 2]

    ax.scatter3D(xs_r[:300], ys_r[:300], zs_r[:300], c='r', marker='o', s=3, alpha=0.6)
    ax.scatter3D(xs_r[300:], ys_r[300:], zs_r[300:], c='b', marker='o', s=3, alpha=0.6)
    plt.legend(['Sim', 'Real'])

    plt.show()


if __name__ == "__main__":
    """
    Need to provide 3 parameters in list below:
        file: Path to trained policy (e.g.: /path/to/params.pkl)
        vae: Path to trained adapted VAE (e.g.: /path/to/ckpt.pth) 
        path_to_images_r, path_to_images_s: Path to data want to plot (in main() function)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--vae', type=str, default=None, help='path to the vae real checkpoint')
    parser.add_argument('--mode', default='video_env', type=str, help='env mode')
    parser.add_argument('--gpu', action='store_false')
    args = parser.parse_args()
    main(args)
