import os
import glob
import argparse
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
from tqdm import tqdm
import datetime
import dateutil.tz

import torch
import torch.nn.functional as F
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.torch.vae.conv_vae import (ConvVAE,)
from rlkit.pythonplusplus import identity
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core import logger


def create_target_encoder(vae_kwargs, representation_size, decoder_activation):
    conv_vae = ConvVAE(
        representation_size,
        decoder_output_activation=decoder_activation,
        **vae_kwargs
    )
    return conv_vae


def load_real_n_sim_data(path=''):
    sim_images, real_images = [], []
    if os.path.exists(path):
        sim_filenames = glob.glob(os.path.join(path, 'images_sim', '*.npz'))
        real_filenames = glob.glob(os.path.join(path, 'images_real', '*.npz'))
        sim_filenames.sort()
        real_filenames.sort()
        assert len(sim_filenames) == len(real_filenames), "Number of data are not equal"
        N = len(sim_filenames)
        for i in range(N):
            im_sim = np.load(sim_filenames[i])
            im_real = np.load(real_filenames[i])
            sim_images.append(im_sim['im'])
            real_images.append(im_real['im'])
    else:
        print('[WARNING] Cannot find data')

    return np.array(sim_images), np.array(real_images)


def main(args):
    # User defined parameter
    vae_kwargs = dict(
        input_channels=3,
        architecture=imsize48_default_architecture,
        decoder_distribution='gaussian_identity_variance',
    )
    representation_size = 4
    decoder_activation = identity
    path_to_data = '/mnt/hdd/tung/workspace/rlkit/tester'   # Must have 'images_sim' & 'images_real'
    n_epochs = 100
    step = 50   # Step to decay learning rate
    batch_size = 50
    beta = 20
    alpha = 1.0

    # Load data
    data = torch.load(open(args.file, "rb"))
    source_data, target_data = load_real_n_sim_data(path_to_data)

    # Config for running-time
    if args.gpu:
        ptu.set_gpu_mode(True)

    # Load models
    env = data['evaluation/env']
    src_vae = env.vae
    tgt_vae = create_target_encoder(vae_kwargs, representation_size, decoder_activation)
    src_vae.to(ptu.device)
    tgt_vae.to(ptu.device)

    # Setup criterion and optimizer
    params = tgt_vae.parameters()
    optimizer = optim.Adam(params, lr=1e-2, weight_decay=0,)
    model_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.1)

    # Logger
    default_path = '/mnt/hdd/tung/workspace/rlkit/data/vae_adapt'
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
    prefix_dir = 'vae-exp-date-' + timestamp[:10]
    sub_dir = 'vae-exp-time-' + timestamp[11:19].replace(':', '-')
    if not os.path.exists(os.path.join(default_path, prefix_dir)):
        os.mkdir(os.path.join(default_path, prefix_dir))
    os.mkdir(os.path.join(default_path, prefix_dir, sub_dir))
    ckpt_path = os.path.join(default_path, prefix_dir, sub_dir)
    eval_statistics = OrderedDict()
    tensor_board = SummaryWriter(ckpt_path)
    logger.set_snapshot_dir(ckpt_path)
    logger.add_tabular_output(
        'vae_progress.csv', relative_to_snapshot_dir=True
    )

    n_batchs = len(source_data) // batch_size
    for epoch in tqdm(range(n_epochs)):
        losses, log_probs, kles, pair_losses = [], [], [], []
        idxes = np.random.permutation(len(source_data))
        model_lr_scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        print("Training epoch: %s,  learning rate: %s" % (epoch, cur_lr))
        # ======= Train for each epoch
        tgt_vae.train()
        for b in range(n_batchs):
            images_src = ptu.from_numpy(source_data[idxes[b * batch_size:(b + 1) * batch_size]])
            images_tgt = ptu.from_numpy(target_data[idxes[b * batch_size:(b + 1) * batch_size]])

            src_latent_mean, _ = src_vae.encode(images_src)
            optimizer.zero_grad()
            tgt_reconstruction, tgt_obs_distr_params, tgt_latent_distr_param = tgt_vae(images_tgt)
            log_prob = tgt_vae.logprob(images_tgt, tgt_obs_distr_params)
            kle = tgt_vae.kl_divergence(tgt_latent_distr_param)

            # Compute loss scope
            vae_loss = -1 * log_prob + beta * kle
            pair_loss = F.mse_loss(tgt_latent_distr_param[0], src_latent_mean)
            loss = vae_loss + alpha * pair_loss
            loss.backward()

            losses.append(loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())
            pair_losses.append(pair_loss.item())
            optimizer.step()

        # ======= Test for each epoch
        tgt_vae.eval()
        debug_batch_size = 32
        idx = np.random.randint(0, len(source_data), batch_size)
        images_tgt = ptu.from_numpy(target_data[idx])
        reconstructions, _, _ = tgt_vae(images_tgt)
        img = images_tgt[0]

        recon_mse = ((reconstructions[0] - img) ** 2).mean().view(-1)
        img_repeated = img.expand((debug_batch_size, img.shape[0]))

        samples = ptu.randn(debug_batch_size, representation_size)
        random_imgs, _ = tgt_vae.decode(samples)
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
        save_dict = {
            'epoch': epoch,
            'model': tgt_vae.state_dict()
        }
        torch.save(save_dict, os.path.join(ckpt_path, 'ckpt.pth'))

        eval_statistics['train/Log Prob'] = np.mean(log_probs)
        eval_statistics['train/KL'] = np.mean(kles)
        eval_statistics['train/loss'] = np.mean(losses)
        eval_statistics['train/pair_loss'] = np.mean(pair_losses)
        for key, value in stats.items():
            eval_statistics[key] = value

        for k, v in eval_statistics.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()

        tensor_board.add_scalar('train/log_prob', np.mean(log_probs), epoch)
        tensor_board.add_scalar('train/KL', np.mean(kles), epoch)
        tensor_board.add_scalar('train/loss', np.mean(losses), epoch)
        tensor_board.add_scalar('train/pair_loss', np.mean(pair_losses), epoch)
        for key, value in stats.items():
            tensor_board.add_scalar(key, value, epoch)


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='path to the snapshot file')
parser.add_argument('--checkpoint', type=str, help='path save checkpoint')
parser.add_argument('--gpu', action='store_false')
args = parser.parse_args()


if __name__ == '__main__':
    main(args)
