import argparse
from collections import OrderedDict
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch import optim

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.pythonplusplus import identity
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core import logger
from rlkit.sim2real.sim2real_utils import *
from rlkit.sim2real.discriminator import *
from rlkit.sim2real.generator import *


def adaptation_loss():
    pass


def alignment_loss():
    pass


def get_batch(data, batch_idx, idxes, batch_size):
    batch_data = ptu.from_numpy(data[idxes[batch_idx * batch_size:(batch_idx + 1) * batch_size]])
    return batch_data


def start_epoch(n_train_data, learning_rate_scheduler=None):
    losses_tgt, losses_cri, acces = [], [], []
    idxes = np.random.permutation(n_train_data)
    if learning_rate_scheduler is not None:
        learning_rate_scheduler.step()
    return (losses_tgt, losses_cri, acces), idxes


def finish_epoch():
    pass


def main(args):
    # Config for running-time
    if args.gpu:
        ptu.set_gpu_mode(True)

    ##########################
    # User defined parameter #
    ##########################
    path_to_data = '/mnt/hdd/tung/workspace/rlkit/tester'   # Must have 'images_sim' & 'images_real'
    vae_kwargs = dict(
        input_channels=3,
        architecture=imsize48_default_architecture,
        decoder_distribution='gaussian_identity_variance',
    )
    decoder_activation = identity
    representation_size = 4
    d_hidden_dims = 3
    d_output_dims = 2
    n_epochs = 1000
    step1, step2 = 5000, 5000   # Step to decay learning rate
    batch_size = 64
    beta = 10
    lr = 1e-3
    alpha = 2.0
    # default_path = '/mnt/hdd/tung/workspace/rlkit/data/vae_adapt'
    default_path = '/mnt/hdd/tung/workspace/rlkit/data/gan_adapt'

    ##########################
    #       Load data        #
    ##########################
    rand_src_train, rand_tgt_train, rand_src_eval, rand_tgt_eval, \
    pair_src_train, pair_src_eval, pair_tgt_train, pair_tgt_eval = load_all_required_data(path_to_data)
    print('Number of random data: Sim={}, Real={}'.format(len(rand_src_train), len(rand_tgt_train)))
    print('Number of pair data: Sim={}, Real={}'.format(len(pair_src_train), len(pair_tgt_train)))

    ##########################
    #       Load models      #
    ##########################
    policy_ckpt = torch.load(open(args.file, "rb"))
    env = policy_ckpt['evaluation/env']

    src_vae = env.vae
    # tgt_vae = create_target_encoder(vae_kwargs, representation_size, decoder_activation)
    tgt_vae = Encoder(representation_size)
    critic = Discriminator(representation_size, d_hidden_dims, d_output_dims)

    src_vae.to(ptu.device)
    tgt_vae.to(ptu.device)
    critic.to(ptu.device)
    if args.vae is not None:
        vae_ckpt = torch.load(open(os.path.join(args.vae, 'vae_ckpt.pth'), "rb"))
        tgt_vae.load_state_dict(vae_ckpt)

    # Setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_vae.encoder.parameters(), lr=lr, weight_decay=0,)
    optimizer_cri = optim.Adam(critic.parameters(), lr=lr, weight_decay=0,)
    model_lr_scheduler = None
    # model_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step1, gamma=0.5)
    # model_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_tgt, milestones=[step1, step2], gamma=0.5)

    # Logger
    ckpt_path = setup_logger_path(args=args, path=default_path)
    tensor_board = SummaryWriter(ckpt_path)
    logger.set_snapshot_dir(ckpt_path)
    logger.add_tabular_output('vae_progress.csv', relative_to_snapshot_dir=True)

    n_batchs = len(rand_src_train) // batch_size
    n_train_data = len(rand_src_train)

    # ===== Phase 1: Training for domain adaptation
    for epoch in tqdm(range(n_epochs)):
        (losses_tgt, losses_cri, acces), idxes = \
            start_epoch(n_train_data, model_lr_scheduler)
        # ======= Train for each epoch
        tgt_vae.train()
        critic.train()
        for b in range(n_batchs):
            ###########################
            #   Train discriminator   #
            ###########################
            images_src = get_batch(rand_src_train, b, idxes, batch_size)
            images_tgt = get_batch(rand_tgt_train, b, idxes, batch_size)

            # zero gradients for optimizer
            optimizer_cri.zero_grad()

            # extract and concat features
            src_latent_mean, _ = src_vae.encode(images_src)
            tgt_latent_mean = tgt_vae(images_tgt)
            feat_concat = torch.cat((src_latent_mean, tgt_latent_mean), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.ones(src_latent_mean.size(0)).long().to(ptu.device)
            label_tgt = torch.zeros(tgt_latent_mean.size(0)).long().to(ptu.device)
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_cri.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            #   Train target encoder   #
            ############################

            # zero gradients for optimizer
            optimizer_cri.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            tgt_latent_mean = tgt_vae(images_tgt)

            # predict on discriminator
            pred_tgt = critic(tgt_latent_mean)

            # prepare fake labels
            label_tgt = torch.ones(tgt_latent_mean.size(0)).long().to(ptu.device)

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            losses_tgt.append(loss_critic.item())
            losses_cri.append(loss_tgt.item())
            acces.append(acc.item())

        # ======= Test for each epoch
        tensor_board.add_scalar('train/d_loss', np.mean(losses_cri), epoch)
        tensor_board.add_scalar('train/g_loss', np.mean(losses_tgt), epoch)
        tensor_board.add_scalar('train/accuracy', np.mean(acces), epoch)
        save_dict = {
            'epoch': epoch,
            'model': tgt_vae.state_dict()
        }
        torch.save(save_dict, os.path.join(ckpt_path, 'vae_ckpt.pth'))


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='path to the snapshot file')
parser.add_argument('--vae', type=str, default=None, help='path save checkpoint')
parser.add_argument('--checkpoint', type=str, help='path save checkpoint')
parser.add_argument('--exp', type=str, default=None, help='Another description for experiment name')
parser.add_argument('--gpu', action='store_false')
args_user = parser.parse_args()


if __name__ == '__main__':
    main(args_user)
