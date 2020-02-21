import os
import argparse
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.pythonplusplus import identity
from rlkit.core.logging import MyEncoder
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core import logger
from rlkit.sim2real.sim2real_utils import *


def get_batch(data, batch_idx, idxes, batch_size):
    _n_batchs = len(idxes) // batch_size
    if batch_idx >= _n_batchs:
        batch_idx = batch_idx % _n_batchs
        print('[DEBUG]: Turn over batch at batch_idx={}'.format(batch_idx))
    batch_data = ptu.from_numpy(data[idxes[batch_idx * batch_size:(batch_idx + 1) * batch_size]])
    return batch_data


def start_epoch(n_train_data, learning_rate_scheduler=None):
    losses, log_probs, kles, pair_losses = [], [], [], []
    idxes = np.random.permutation(n_train_data)
    if learning_rate_scheduler is not None:
        learning_rate_scheduler.step()
    return (losses, log_probs, kles, pair_losses), idxes


def setup_logger(args, variant):
    ckpt_path = setup_logger_path(args=args, path=variant['default_path'])
    tensor_board = SummaryWriter(ckpt_path)
    with open(os.path.join(ckpt_path, 'log_params.json'), "w") as f:
        json.dump(variant, f, indent=2, sort_keys=True, cls=MyEncoder)
    eval_statistics = OrderedDict()
    logger.set_snapshot_dir(ckpt_path)
    logger.add_tabular_output(
        'vae_progress.csv', relative_to_snapshot_dir=True
    )

    return eval_statistics, tensor_board, ckpt_path


def pairwise_loss_schedule(value, epoch):
    if epoch < 500:
        return 0.0
    else:
        return value


def main(args):
    # Config for running-time
    if args.gpu:
        ptu.set_gpu_mode(True)

    ##########################
    # User defined parameter #
    ##########################
    variant = dict(
        # Must have 'images_sim' & 'images_real'
        path_to_data='/mnt/hdd/tung/workspace/rlkit/tester',
        vae_kwargs=dict(
            input_channels=3,
            architecture=imsize48_default_architecture,
            decoder_distribution='gaussian_identity_variance',
        ),
        decoder_activation=identity,
        representation_size=4,
        n_epochs=2000,
        step1=5000,  # Step to decay learning rate
        step2=5000,
        batch_size=50,
        lr=1e-3,
        beta=20,
        alpha=1.0,
        # default_path = '/mnt/hdd/tung/workspace/rlkit/data/vae_adapt',
        default_path='/mnt/hdd/tung/workspace/rlkit/data/vae_adapt',
    )


    # Load data
    ##########################
    #       Load data        #
    ##########################
    rand_src_train, rand_tgt_train, rand_src_eval, rand_tgt_eval, \
    pair_src_train, pair_src_eval, pair_tgt_train, pair_tgt_eval = \
        load_all_required_data(variant['path_to_data'])
    print('No. of random train data: Sim={}, Real={}'.format(len(rand_src_train),
                                                             len(rand_tgt_train)))
    print(
        'No. of pair train data: Sim={}, Real={}'.format(len(pair_src_train), len(pair_tgt_train)))

    ##########################
    #       Load models      #
    ##########################
    data = torch.load(open(args.file, "rb"))
    env = data['evaluation/env']
    src_vae = env.vae

    tgt_vae = create_target_encoder(variant['vae_kwargs'],
                                    variant['representation_size'],
                                    variant['decoder_activation'])
    src_vae.to(ptu.device)
    tgt_vae.to(ptu.device)

    # Setup criterion and optimizer
    params = tgt_vae.parameters()
    optimizer = optim.Adam(params, lr=variant['lr'])
    model_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=variant['step1'], gamma=0.1)

    # Logger
    eval_statistics, tensor_board, ckpt_path = setup_logger(args, variant)

    n_batchs = len(rand_src_train) // variant['batch_size']
    n_train_rand = len(rand_src_train)
    n_train_pair = len(pair_src_train)

    # Training for VAE
    for epoch in tqdm(range(variant['n_epochs'])):
        (losses, log_probs, kles, pair_losses), idxes_rand_data = \
            start_epoch(n_train_rand, model_lr_scheduler)
        _, idxes_pair_data = start_epoch(n_train_pair)
        # ======= Train for each epoch
        tgt_vae.train()

        for b in range(n_batchs):
            rand_img_tgt = get_batch(rand_tgt_train, b, idxes_rand_data, variant['batch_size'])
            pair_img_src = get_batch(pair_src_train, b, idxes_pair_data, variant['batch_size'])
            pair_img_tgt = get_batch(pair_tgt_train, b, idxes_pair_data, variant['batch_size'])

            src_latent_mean, _ = src_vae.encode(pair_img_src)
            optimizer.zero_grad()
            tgt_reconstruction, tgt_obs_distr_params, tgt_latent_distr_param = tgt_vae(rand_img_tgt)
            log_prob = tgt_vae.logprob(rand_img_tgt, tgt_obs_distr_params)
            kle = tgt_vae.kl_divergence(tgt_latent_distr_param)

            # VAE loss
            vae_loss = -1 * log_prob + variant['beta'] * kle

            # Pairwise loss
            _, _, pair_tgt_latent_distr_param = tgt_vae(pair_img_tgt)
            pair_loss = F.mse_loss(pair_tgt_latent_distr_param[0], src_latent_mean)

            loss = vae_loss + variant['alpha'] * pair_loss
            loss.backward()

            losses.append(loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())
            pair_losses.append(pair_loss.item())
            optimizer.step()

        # ======= Test for each epoch
        tgt_vae.eval()
        debug_batch_size = 32
        idx = np.random.randint(0, len(rand_tgt_eval), variant['batch_size'])
        images_tgt = ptu.from_numpy(rand_tgt_eval[idx])
        reconstructions, _, _ = tgt_vae(images_tgt)
        img = images_tgt[0]

        recon_mse = ((reconstructions[0] - img) ** 2).mean().view(-1)
        img_repeated = img.expand((debug_batch_size, img.shape[0]))

        samples = ptu.randn(debug_batch_size, variant['representation_size'])
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
        torch.save(save_dict, os.path.join(ckpt_path, 'vae_ckpt.pth'))

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
parser.add_argument('--vae', type=str, default=None, help='path save checkpoint')
parser.add_argument('--exp', type=str, default=None, help='Another description for experiment name')
parser.add_argument('--gpu', action='store_false')
args = parser.parse_args()


if __name__ == '__main__':
    main(args)
