import os
import argparse
import numpy as np

from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in_aim_v1

from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.pythonplusplus import identity
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.sim2real.sim2real_utils import *
from rlkit.sim2real.goal_test import set_of_goals
from rlkit.samplers.rollout_functions import multitask_rollout_sim2real


def main(args):
    # Config for running-time
    if args.gpu:
        ptu.set_gpu_mode(True)

    ##########################
    # User defined parameter #
    ##########################
    variant = dict(
        seed=None,
        path_to_data='/mnt/hdd/tung/workspace/rlkit/tester',
        rand_src_dir='rand_img_sim_tgt_new.10000',
        rand_tgt_dir='rand_img_sim_tgt_new.10000',

        # pair_src_dir='randlarger_pair_sim_src.1000',
        # pair_tgt_dir='randlarger_pair_sim_tgt.1000',
        # pair_src_dir='randlarger_pair_sim_src.2000',
        # pair_tgt_dir='randlarger_pair_sim_tgt.2000',
        # pair_src_dir='randlarger_pair_sim_src.3000',
        # pair_tgt_dir='randlarger_pair_sim_tgt.3000',
        pair_src_dir='rand_pair_sim_src.5000',
        pair_tgt_dir='rand_pair_sim_tgt.5000',
        test_ratio=0.1,
        vae_kwargs=dict(
            input_channels=3,
            architecture=imsize48_default_architecture,
            decoder_distribution='gaussian_identity_variance',
            # decoder_distribution='laplace_identity_variance',
            # decoder_distribution='huber_identity_variance',
        ),
        decoder_activation=identity,
        representation_size=4,
        # default_path='/mnt/hdd/tung/workspace/rlkit/data/vae_adapt_10000rand_1000pair/',
        # default_path='/mnt/hdd/tung/workspace/rlkit/data/vae_adapt_10000rand_2000pair/',
        # default_path='/mnt/hdd/tung/workspace/rlkit/data/vae_adapt_10000rand_3000pair/',
        default_path='/mnt/hdd/tung/workspace/rlkit/data/vae_adapt_10000rand_5000pair_old/',
        beta=20,

        # TUNG: REGION FOR TRAINING
        vae_loss_opt=dict(
            loss_type='beta_vae_loss',
            # vae_loss='beta_vae_loss_rm_rec',
            # vae_loss='beta_vae_loss_stop_grad_dec',
            beta=20,
        ),
        da_loss_opt=dict(
            loss_type='consistency_loss',
            # loss_type='consistency_loss_w_cycle',
            # loss_type='consistency_loss_rm_dec_path',

            ctc_latent_cross=False,

            alpha1=1.0,
            alpha2=0.0,
            alpha3=0.0,

            use_mu=False,

            distance=mse_loss,
            # distance=l1_loss,
            # distance=huber_loss,
        ),
        init_tgt_by_src=True,
        alpha0=1.0,

        n_epochs=2000,
        step1=10000,  # Step to decay learning rate
        batch_size=50,
        lr=1e-3,

        # TUNG: REGION FOR TESTING
        test_opt=dict(
            test_enable=True,
            period=50,
            env_name='SawyerPushNIPSCustomEasy-v0',
            imsize=48,
            init_camera=sawyer_init_camera_zoomed_in_aim_v1,
            n_test=2,
            H=50,
            hide=True
        )
    )
    if args.seed is None:
        variant['seed'] = random.randint(0, 100000)
    set_seed(variant['seed'])

    ##########################
    #       Load data        #
    ##########################
    print('[INFO] Loading data...')
    rand_src_train, rand_tgt_train, rand_src_eval, rand_tgt_eval, \
    pair_src_train, pair_src_eval, pair_tgt_train, pair_tgt_eval = \
        load_all_required_data(variant['path_to_data'],
                               variant['rand_src_dir'], variant['rand_tgt_dir'],
                               variant['pair_src_dir'], variant['pair_tgt_dir'],
                               test_ratio=variant['test_ratio'])
    print('Rand train data: Sim={}, Real={}'.format(len(rand_src_train), len(rand_tgt_train)))
    print('Pair train data: Sim={}, Real={}'.format(len(pair_src_train), len(pair_tgt_train)))

    ##########################
    #       Load models      #
    ##########################
    print('[INFO] Loading checkpoint...')
    data = torch.load(open(args.file, "rb"))
    env = data['evaluation/env']
    policy = data['evaluation/policy']
    src_vae = env.vae

    tgt_vae = create_target_encoder(variant['vae_kwargs'],
                                    variant['representation_size'],
                                    variant['decoder_activation'])
    src_vae.to(ptu.device)
    tgt_vae.to(ptu.device)

    if variant['init_tgt_by_src']:
        tgt_vae.load_state_dict(src_vae.state_dict())

    # Setup for testing
    test_env = create_vae_wrapped_env(env_name=variant['test_opt']['env_name'],
                                      vae=tgt_vae,
                                      imsize=variant['test_opt']['imsize'],
                                      init_camera=variant['test_opt']['init_camera'])

    # Setup criterion and optimizer
    params = tgt_vae.parameters()
    optimizer = optim.Adam(params, lr=variant['lr'])
    model_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=variant['step1'], gamma=0.1)

    # Logger
    eval_statistics, tensor_board, ckpt_path = setup_logger(args, variant)

    n_batchs = len(rand_src_train) // variant['batch_size']
    n_train_rand = len(rand_src_train)
    n_train_pair = len(pair_src_train)

    vaeloss = vae_loss_option(variant['vae_loss_opt'])
    daloss = da_loss_option(variant['da_loss_opt'])

    # ======================= TRAINING =======================
    for epoch in tqdm(range(variant['n_epochs'])):
        (losses, log_probs, kles, pair_losses), idxes_rand_data = \
            start_epoch(n_train_rand, model_lr_scheduler)
        _, idxes_pair_data = start_epoch(n_train_pair)

        tgt_vae.train()
        for b in range(n_batchs):
            rand_img_tgt = get_batch(rand_tgt_train, b, idxes_rand_data, variant['batch_size'])

            pair_img_src = get_batch(pair_src_train, b, idxes_pair_data, variant['batch_size'])
            pair_img_tgt = get_batch(pair_tgt_train, b, idxes_pair_data, variant['batch_size'])

            optimizer.zero_grad()

            # VAE loss
            total_vae_loss, log_prob, kle = vaeloss(tgt_vae, rand_img_tgt, variant['vae_loss_opt'])

            # Pairwise loss
            pair_loss = daloss(pair_img_src, pair_img_tgt, src_vae, tgt_vae, variant['da_loss_opt'])

            loss = variant['alpha0'] * total_vae_loss + pair_loss
            loss.backward()

            losses.append(loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())
            pair_losses.append(pair_loss.item())
            optimizer.step()

        # ======= Test for each epoch
        end_epoch(tgt_net=tgt_vae, src_net=src_vae,
                  rand_data_real_eval=rand_tgt_eval, rand_data_sim_eval=rand_src_eval,
                  pair_data_real_eval=pair_tgt_eval, pair_data_sim_eval=pair_src_eval,
                  statistics=eval_statistics, tb=tensor_board,
                  variant=variant, save_path=ckpt_path, epoch=epoch,
                  losses=losses, log_probs=log_probs, kles=kles, pair_losses=pair_losses,
                  test_env=test_env, policy=policy)


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='path to the snapshot file')
parser.add_argument('--vae', type=str, default=None, help='path save checkpoint')
parser.add_argument('--exp', type=str, default=None, help='Another description for experiment name')
parser.add_argument('--gpu', action='store_false')
parser.add_argument('--seed', type=int, default=None)
user_args = parser.parse_args()


if __name__ == '__main__':
    main(user_args)
