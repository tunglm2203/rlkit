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


def start_epoch(n_train_data, learning_rate_scheduler=None):
    losses, log_probs, kles, pair_losses = [], [], [], []
    idxes = np.random.permutation(n_train_data)
    if learning_rate_scheduler is not None:
        learning_rate_scheduler.step()
    return (losses, log_probs, kles, pair_losses), idxes


def end_epoch(tgt_net, src_net,
              rand_data_real_eval, rand_data_sim_eval, pair_data_real_eval, pair_data_sim_eval,
              statistics, tb, variant, save_path, epoch,
              losses, log_probs, kles, pair_losses, test_env, policy):
    # ============== Test for debugging VAE ==============
    tgt_net.eval()
    debug_batch_size = 32
    idx = np.random.randint(0, len(rand_data_real_eval), variant['batch_size'])
    images = ptu.from_numpy(rand_data_real_eval[idx])
    reconstructions, _, _ = tgt_net(images)
    img = images[0]

    recon_mse = ((reconstructions[0] - img) ** 2).mean().view(-1)
    img_repeated = img.expand((debug_batch_size, img.shape[0]))

    samples = ptu.randn(debug_batch_size, variant['representation_size'])
    random_imgs, _ = tgt_net.decode(samples)
    random_mses = (random_imgs - img_repeated) ** 2
    mse_improvement = ptu.get_numpy(random_mses.mean(dim=1) - recon_mse)
    # ============== Test for debugging VAE ==============

    # ========= Test for debugging pairwise data =========
    eval_batch_size = 100
    idxes = np.arange(len(pair_data_real_eval))
    latent_pair_loss = []
    for b in range(len(pair_data_real_eval) // eval_batch_size):
        eval_sim = get_batch(pair_data_sim_eval, b, idxes, eval_batch_size)
        eval_real = get_batch(pair_data_real_eval, b, idxes, eval_batch_size)
        latent_sim, _ = src_net.encode(eval_sim)
        latent_real, _ = tgt_net.encode(eval_real)
        eval_loss = F.mse_loss(latent_real, latent_sim)
        latent_pair_loss.append(eval_loss.item())
    # ========= Test for debugging pairwise data =========

    # ========= Test adapted VAE in target domain env =========
    if variant['test_opt']['test_enable'] and \
            (epoch % variant['test_opt']['period'] == 0 or epoch == variant['n_epochs'] - 1):
        ob_dist_mean, ob_dist_std, ee_dist_mean, ee_dist_std = \
            rollout_pusher(variant, policy, test_env)

    stats = create_stats_ordered_dict('debug/MSE improvement over random', mse_improvement)
    stats.update(create_stats_ordered_dict('debug/MSE of random decoding',
                                           ptu.get_numpy(random_mses)))
    stats['debug/MSE of reconstruction'] = ptu.get_numpy(recon_mse)[0]
    save_dict = {
        'epoch': epoch,
        'model': tgt_net.state_dict()
    }
    torch.save(save_dict, os.path.join(save_path, 'vae_ckpt.pth'))

    statistics['train/Log Prob'] = np.mean(log_probs)
    statistics['train/KL'] = np.mean(kles)
    statistics['train/loss'] = np.mean(losses)
    statistics['train/pair_loss'] = np.mean(pair_losses)
    statistics['eval/pair_loss'] = np.mean(latent_pair_loss)
    for key, value in stats.items():
        statistics[key] = value
    for k, v in statistics.items():
        logger.record_tabular(k, v)
    logger.dump_tabular()

    tb.add_scalar('train/log_prob', np.mean(log_probs), epoch)
    tb.add_scalar('train/KL', np.mean(kles), epoch)
    tb.add_scalar('train/loss', np.mean(losses), epoch)
    tb.add_scalar('train/pair_loss', np.mean(pair_losses), epoch)
    tb.add_scalar('eval/pair_loss', np.mean(latent_pair_loss), epoch)
    if variant['test_opt']['test_enable'] and \
            (epoch % variant['test_opt']['period'] == 0 or epoch == variant['n_epochs'] - 1):
        tb.add_scalar('eval/puck_distance (mean)', ob_dist_mean, epoch)
        tb.add_scalar('eval/puck_distance (std)', ob_dist_mean, epoch)
        tb.add_scalar('eval/hand_distance (mean)', ee_dist_mean, epoch)
        tb.add_scalar('eval/hand_distance (std)', ee_dist_std, epoch)

    for key, value in stats.items():
        tb.add_scalar(key, value, epoch)


def rollout_pusher(variant, policy, env):
    n_goals = len(set_of_goals)
    final_puck_distance = np.zeros((n_goals, variant['test_opt']['n_test']))
    final_hand_distance = np.zeros_like(final_puck_distance)
    print('[INFO] Starting test in set of goals...')
    for goal_id in tqdm(range(n_goals)):
        # Assign goal from set of goals
        env.wrapped_env.wrapped_env.fixed_hand_goal = set_of_goals[goal_id][0]
        env.wrapped_env.wrapped_env.fixed_puck_goal = set_of_goals[goal_id][1]

        # Number of test per each goal
        paths_per_goal = []
        for i in range(variant['test_opt']['n_test']):
            paths_per_goal.append(multitask_rollout_sim2real(
                env,
                policy,
                max_path_length=variant['test_opt']['H'],
                render=not variant['test_opt']['hide'],
                observation_key='observation',
                desired_goal_key='desired_goal',
            ))
            final_puck_distance[goal_id, i] = paths_per_goal[i]['env_infos'][-1]['puck_distance']
            final_hand_distance[goal_id, i] = paths_per_goal[i]['env_infos'][-1]['hand_distance']

    ob_dist_mean = final_puck_distance.mean()
    ob_dist_std = final_puck_distance.mean(axis=0).std()
    ee_dist_mean = final_hand_distance.mean()
    ee_dist_std = final_hand_distance.mean(axis=0).std()
    return ob_dist_mean, ob_dist_std, ee_dist_mean, ee_dist_std


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
        pair_src_dir='randlarger_pair_sim_src.3000',
        pair_tgt_dir='randlarger_pair_sim_tgt.3000',
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
        default_path='/mnt/hdd/tung/workspace/rlkit/data/vae_adapt_10000rand_3000pair/',
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
