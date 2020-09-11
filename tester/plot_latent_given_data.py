import argparse
import glob
import os
import time
import copy
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import cv2

from multiworld.core.image_env import unormalize_image

import torch

from rlkit.core import logger
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.sim2real.generator import *
from rlkit.sim2real.sim2real_utils import *


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


def debug_learning_curve_learned_vae(args,
                                     path_to_images='/mnt/hdd/tung/workspace/rlkit/tester/images_sim'):
    # Load images
    print('Loading images...')
    images = load_data(path=path_to_images)
    # Load pre-trained policy
    print("Loading policy and environment...")
    data = torch.load(open(args.file, "rb"))
    env = data['evaluation/env']

    # Get hyper-parameters
    representation_size = data['evaluation/env'].representation_size

    # User hyper-parameters
    n_epochs = 200
    path_to_save_debug = './debug'
    imsize = 48
    beta = 20

    if args.gpu:
        ptu.set_gpu_mode(True)
    if isinstance(env, VAEWrappedEnv) and hasattr(env, 'mode'):
        env.mode(args.mode)
    vae = env.vae  # Get learned VAE from env

    if args.vae is not None:
        adapted_vae = torch.load(open(os.path.join(args.vae, 'vae_ckpt.pth'), "rb"))
        vae.load_state_dict(adapted_vae['model'])

    vae.eval()
    # Logger
    eval_statistics = OrderedDict()
    logger.set_snapshot_dir(path_to_save_debug)
    logger.add_tabular_output(
        'vae_progress.csv', relative_to_snapshot_dir=True
    )

    # ======= Test for each epoch
    for epoch in range(n_epochs):
        debug_batch_size = 32
        images_batch = ptu.from_numpy(images)
        reconstructions, obs_dist_params, latent_dist_params = vae(images_batch)
        img = images_batch[0]

        log_prob = vae.logprob(images_batch, obs_dist_params)
        kle = vae.kl_divergence(latent_dist_params)
        loss = -1 * log_prob + beta * kle

        recon_mse = ((reconstructions[0] - img) ** 2).mean().view(-1)
        img_repeated = img.expand((debug_batch_size, img.shape[0]))

        samples = ptu.randn(debug_batch_size, representation_size)
        random_imgs, _ = vae.decode(samples)
        random_mses = (random_imgs - img_repeated) ** 2
        mse_improvement = ptu.get_numpy(random_mses.mean(dim=1) - recon_mse)

        stats = create_stats_ordered_dict(
            'debug/MSE improvement over random',
            mse_improvement,
        )
        stats.update(create_stats_ordered_dict(
            'debug/MSE of random decoding',
            ptu.get_numpy(random_mses),
        ))
        stats['debug/MSE of reconstruction'] = ptu.get_numpy(
            recon_mse
        )[0]
        eval_statistics['train/Log Prob'] = log_prob.item()
        eval_statistics['train/KL'] = kle.item()
        eval_statistics['train/loss'] = loss.item()
        eval_statistics['train/pair_loss'] = None
        for key, value in stats.items():
            eval_statistics[key] = value

        for k, v in eval_statistics.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()


def main(args):
    # ===================== USER SCOPE =====================
    print('Loading images...')
    root_path = '/mnt/hdd/tung/workspace/rlkit/tester/'

    rand_src_dir = None
    rand_tgt_dir = None
    # pair_src_dir = 'images_sim'
    # pair_tgt_dir = 'images_real'
    # pair_src_dir = 'random_pair_sim'
    # pair_tgt_dir = 'random_pair_real'
    # pair_src_dir = 'rand_pair_sim_src.500'
    # pair_tgt_dir = 'rand_pair_sim_tgt.500'
    pair_src_dir = 'rand_pair_sim_src.1000'
    pair_tgt_dir = 'rand_pair_sim_tgt.1000'

    # ===================== LOAD & CREATE MODEL SCOPE =====================
    _, _, _, _, pair_src_train, pair_src_eval, pair_tgt_train, pair_tgt_eval = \
        load_all_required_data(root_path, rand_src_dir, rand_tgt_dir, pair_src_dir, pair_tgt_dir)
    pair_im_src = np.concatenate((pair_src_train, pair_src_eval))
    pair_im_tgt = np.concatenate((pair_tgt_train, pair_tgt_eval))

    print('[INFO] Number of real images: ', len(pair_im_tgt))
    print('[INFO] Number of sim images: ', len(pair_im_src))

    # Load pre-trained policy
    print("[INFO] Loading policy, environment, VAE ...")
    sim_data = torch.load(open(args.file, "rb"))
    env = sim_data['evaluation/env']

    if args.gpu:
        ptu.set_gpu_mode(True)
    if isinstance(env, VAEWrappedEnv) and hasattr(env, 'mode'):
        env.mode(args.mode)

    src_vae = env.vae   # Get learned VAE from env

    if args.vae is not None:
        adapted_vae = torch.load(open(os.path.join(args.vae, 'vae_ckpt.pth'), "rb"))
        tgt_vae = copy.deepcopy(src_vae)
        tgt_vae.load_state_dict(adapted_vae['model'])

    if args.gan is not None:
        adapted_gan = torch.load(open(os.path.join(args.gan, 'vae_ckpt.pth'), "rb"))
        tgt_vae = Encoder(representation_size=4)
        tgt_vae.to(ptu.device)
        tgt_vae.load_state_dict(adapted_gan['model'])

    print("[INFO] Load successfully.")

    assert pair_im_tgt.dtype == np.float64 and pair_im_src.dtype == np.float64, "Image wrong format"
    pair_im_tgt_tensor = ptu.from_numpy(pair_im_tgt)
    pair_im_src_tensor = ptu.from_numpy(pair_im_src)

    # ===================== PROCESS DATA TO LATENT SPACE =====================
    if args.vae is not None:
        latent_distribution_params_r, _ = tgt_vae.encode(pair_im_tgt_tensor)
    elif args.gan is not None:
        latent_distribution_params_r = tgt_vae(pair_im_tgt_tensor)
    else:
        latent_distribution_params_r, _ = src_vae.encode(pair_im_tgt_tensor)
    latent_distribution_params_s, _ = src_vae.encode(pair_im_src_tensor)

    latents_r = ptu.get_numpy(latent_distribution_params_r)
    latents_s = ptu.get_numpy(latent_distribution_params_s)
    # Merge sim & real data together to make same relative coordinate
    latents = np.concatenate((latents_s, latents_r), axis=0)

    # ===================== PROJECTING AND PLOTTING =====================
    # T-SNE
    time_start = time.time()
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(latents)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    plt.figure()
    ax = plt.axes(projection='3d')
    xs_r = tsne_results[:, 0]
    ys_r = tsne_results[:, 1]
    zs_r = tsne_results[:, 2]

    n_data = len(pair_im_tgt)
    n_train = len(pair_src_train)

    ax.scatter3D(xs_r[:n_train], ys_r[:n_train], zs_r[:n_train],
                 c='r', marker='o', s=3, alpha=0.6)
    ax.scatter3D(xs_r[n_data:n_data + n_train], ys_r[n_data:n_data + n_train], zs_r[n_data:n_data + n_train],
                 c='b', marker='o', s=3, alpha=0.6)

    if user_args.split:
        plt.legend(['Sim_train', 'Real_train'])
        plt.title('Training')
        plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(xs_r[n_train:n_data], ys_r[n_train:n_data], zs_r[n_train:n_data],
                      c='darkorange', marker='^', s=4, alpha=0.9)
        ax1.scatter3D(xs_r[n_data + n_train:], ys_r[n_data + n_train:], zs_r[n_data + n_train:],
                      c='darkgreen', marker='^', s=4, alpha=0.9)
        plt.legend(['Sim_eval', 'Real_eval'])
        plt.title('Validation')
    else:
        ax.scatter3D(xs_r[n_train:n_data], ys_r[n_train:n_data], zs_r[n_train:n_data],
                     c='darkorange', marker='^', s=4, alpha=0.9)
        ax.scatter3D(xs_r[n_data + n_train:], ys_r[n_data + n_train:], zs_r[n_data + n_train:],
                     c='darkgreen', marker='^', s=4, alpha=0.9)
        plt.legend(['Sim_train', 'Real_train', 'Sim_eval', 'Real_eval'])

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='path to the snapshot file')
parser.add_argument('--vae', type=str, default=None, help='path to the vae real checkpoint')
parser.add_argument('--gan', type=str, default=None, help='path to the vae real checkpoint')
parser.add_argument('--mode', default='video_env', type=str, help='env mode')
parser.add_argument('--gpu', action='store_false')
parser.add_argument('--split', action='store_true')
user_args = parser.parse_args()

if __name__ == "__main__":
    """
    Need to provide 3 parameters in list below:
        file: Path to trained policy (e.g.: /path/to/params.pkl)
        vae: Path to trained adapted VAE (e.g.: /path/to/vae_file, contains vae_ckpt.pth) 
        path_to_images_r, path_to_images_s: Path to data want to plot (in main() function)
    """

    main(user_args)
    # debug_learning_curve_learned_vae(args)

