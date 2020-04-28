import os
import argparse
import numpy as np

from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler

from rlkit.torch.vae.conv_vae import imsize48_default_architecture
from rlkit.pythonplusplus import identity
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.sim2real.sim2real_utils import *


def start_epoch(n_train_data, learning_rate_scheduler=None):
    losses, log_probs, kles, pair_losses = [], [], [], []
    idxes = np.random.permutation(n_train_data)
    if learning_rate_scheduler is not None:
        learning_rate_scheduler.step()
    return (losses, log_probs, kles, pair_losses), idxes


def end_epoch(tgt_net, src_net,
              rand_data_real_eval, rand_data_sim_eval, pair_data_real_eval, pair_data_sim_eval,
              statistics, tb, variant, save_path, epoch,
              losses, log_probs, kles, pair_losses):
    # ============== Test for debugging VAE ==============
    tgt_net.eval()
    debug_batch_size = 64
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
    for key, value in stats.items():
        tb.add_scalar(key, value, epoch)


def vae_loss(vae, batch_data, opt):
    beta = opt.get('beta', 0)
    # _, tgt_obs_distr_params, tgt_latent_distr_param = vae(batch_data)
    tgt_latent_distr_param = vae.encode(batch_data)
    latents = vae.reparameterize(tgt_latent_distr_param)
    _, tgt_obs_distr_params = vae.decode(latents)
    log_prob = vae.logprob(batch_data, tgt_obs_distr_params)
    kle = vae.kl_divergence(tgt_latent_distr_param)
    loss = -1 * log_prob + beta * kle
    return loss, log_prob, kle


def vae_loss_rm_rec(vae, batch_data, opt):
    beta = opt.get('beta', 0)
    log_prob = torch.FloatTensor(([0.0]))
    _, tgt_obs_distr_params, tgt_latent_distr_param = vae(batch_data)
    # log_prob = vae.logprob(batch_data, tgt_obs_distr_params)
    kle = vae.kl_divergence(tgt_latent_distr_param)
    loss = beta * kle
    return loss, log_prob, kle


def vae_loss_stop_grad_dec(vae, batch_data, opt):
    beta = opt.get('beta', 0)
    # _, tgt_obs_distr_params, tgt_latent_distr_param = vae(batch_data)
    tgt_latent_distr_param = vae.encode(batch_data)
    latents = vae.reparameterize(tgt_latent_distr_param)
    with torch.no_grad():
        _, tgt_obs_distr_params = vae.decode(latents)

    log_prob = vae.logprob(batch_data, tgt_obs_distr_params)
    kle = vae.kl_divergence(tgt_latent_distr_param)
    loss = -1 * log_prob + beta * kle
    return loss, log_prob, kle


def consistency_loss(pair_sim, pair_real, sim_vae, real_vae, opt):
    use_mu = opt.get('use_mu', False)
    alpha1 = opt.get('alpha1', 0)

    f_distance = opt.get('distance', None)
    assert f_distance is not None, 'Must specify the distance function'

    latent_distribution_params_sim = sim_vae.encode(pair_sim)
    if use_mu:
        _, obs_distribution_params_real = real_vae.decode(latent_distribution_params_sim[0])
    else:
        latents_sim = sim_vae.reparameterize(latent_distribution_params_sim)
        _, obs_distribution_params_real = real_vae.decode(latents_sim)

    latent_distribution_params_real = real_vae.encode(pair_real)
    if use_mu:
        _, obs_distribution_params_sim = sim_vae.decode(latent_distribution_params_real[0])
    else:
        latents_real = real_vae.reparameterize(latent_distribution_params_real)
        _, obs_distribution_params_sim = sim_vae.decode(latents_real)

    # ctc_sim2real = real_vae.logprob(pair_real, obs_distribution_params_real)
    # ctc_real2sim = sim_vae.logprob(pair_sim, obs_distribution_params_sim)
    ctc_sim2real = f_distance(pair_real, obs_distribution_params_real[0])
    ctc_real2sim = f_distance(pair_sim, obs_distribution_params_sim[0])
    return alpha1 * (ctc_sim2real + ctc_real2sim)


def consistency_loss_w_cycle(pair_sim, pair_real, sim_vae, real_vae, opt):
    ctc_latent_cross = True
    alpha1 = opt.get('alpha1', 0)
    alpha2 = opt.get('alpha2', 0)
    alpha3 = opt.get('alpha3', 0)
    f_distance = opt.get('distance', None)
    assert f_distance is not None, 'Must specify the distance function'
    # ============== Consitency loss of image ==============
    latent_params_sim = sim_vae.encode(pair_sim)
    _, rec_params_real = real_vae.decode(latent_params_sim[0])

    latent_params_real = real_vae.encode(pair_real)
    _, rec_params_sim = sim_vae.decode(latent_params_real[0])

    ctc_sim2real = f_distance(pair_real, rec_params_real[0], imlength=real_vae.imlength)
    ctc_real2sim = f_distance(pair_sim, rec_params_sim[0], imlength=real_vae.imlength)
    ctc_total = ctc_sim2real + ctc_real2sim

    # ============== Cycle loss of image ==============
    rec_latent_params_real = real_vae.encode(rec_params_real[0].detach())
    _, rec_params_sim_2 = sim_vae.decode(rec_latent_params_real[0])

    rec_latent_params_sim = sim_vae.encode(rec_params_sim[0].detach())
    _, rec_params_real_2 = real_vae.decode(rec_latent_params_sim[0])

    cycle_sim2sim = f_distance(pair_sim, rec_params_sim_2[0])
    cycle_real2real = f_distance(pair_real, rec_params_real_2[0])
    cycle_total = cycle_sim2sim + cycle_real2real

    # ============== Consitency loss of latent ==============
    _, rec_params_real_hat = real_vae.decode(latent_params_sim[0].detach())
    latent_params_real_hat = real_vae.encode(rec_params_real_hat[0])
    if ctc_latent_cross:
        ctc_latent_sim2real = f_distance(latent_params_real[0].detach(), latent_params_real_hat[0])
    else:
        ctc_latent_sim2sim = f_distance(latent_params_sim[0].detach(), latent_params_real_hat[0])

    _, obs_params_sim_hat = sim_vae.decode(latent_params_real[0].detach())
    latent_params_sim_hat = sim_vae.encode(obs_params_sim_hat[0])
    if ctc_latent_cross:
        ctc_latent_real2sim = f_distance(latent_params_sim[0].detach(), latent_params_sim_hat[0])
    else:
        ctc_latent_real2real = f_distance(latent_params_real[0].detach(), latent_params_sim_hat[0])

    if ctc_latent_cross:
        ctc_latent_total = ctc_latent_sim2real + ctc_latent_real2sim
    else:
        ctc_latent_total = ctc_latent_sim2sim + ctc_latent_real2real

    total_loss = alpha1 * ctc_total + alpha2 * cycle_total + alpha3 * ctc_latent_total
    return total_loss


def consistency_loss_rm_dec_path(pair_sim, pair_real, sim_vae, real_vae, opt):
    use_mu = opt.get('use_mu', False)
    # latent_distribution_params_sim = sim_vae.encode(pair_sim)
    # if use_mu:
    #     _, obs_distribution_params_real = real_vae.decode(latent_distribution_params_sim[0])
    # else:
    #     latents_sim = sim_vae.reparameterize(latent_distribution_params_sim)
    #     _, obs_distribution_params_real = real_vae.decode(latents_sim)

    latent_distribution_params_real = real_vae.encode(pair_real)
    if use_mu:
        _, obs_distribution_params_sim = sim_vae.decode(latent_distribution_params_real[0])
    else:
        latents_real = real_vae.reparameterize(latent_distribution_params_real)
        _, obs_distribution_params_sim = sim_vae.decode(latents_real)

    # ctc_sim2real = real_vae.logprob(pair_real, obs_distribution_params_real)
    ctc_real2sim = sim_vae.logprob(pair_sim, obs_distribution_params_sim)
    # return -1 * (ctc_sim2real + ctc_real2sim)
    return -1 * ctc_real2sim


def mse_pair(pair_sim, pair_real, sim_vae, real_vae):
    src_latent_mean, _ = sim_vae.encode(pair_sim)
    tgt_latent_mean, _ = real_vae.encode(pair_real)
    loss = F.mse_loss(tgt_latent_mean, src_latent_mean)
    return loss


def vae_loss_option(vae_loss_opt):
    loss_type = vae_loss_opt.get('loss_type', None)
    if loss_type == 'beta_vae_loss':
        return vae_loss
    elif loss_type == 'beta_vae_loss_rm_rec':
        return vae_loss_rm_rec
    elif loss_type == 'beta_vae_loss_stop_grad_dec':
        return vae_loss_stop_grad_dec
    else:
        raise NotImplementedError('VAE loss {} not supported'.format(loss_type))


def da_loss_option(da_loss_opt):
    loss_type = da_loss_opt.get('loss_type', None)
    if loss_type == 'consistency_loss':
        return consistency_loss
    elif loss_type == 'consistency_loss_w_cycle':
        return consistency_loss_w_cycle
    elif loss_type == 'consistency_loss_rm_dec_path':
        return consistency_loss_rm_dec_path
    else:
        raise NotImplementedError('DA loss {} not supported'.format(loss_type))


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
        rand_src_dir='rand_img_sim_tgt.10000',
        rand_tgt_dir='rand_img_sim_tgt.10000',
        pair_src_dir='rand_pair_sim_src.1000',
        pair_tgt_dir='rand_pair_sim_tgt.1000',
        test_ratio=0.2,
        vae_kwargs=dict(
            input_channels=3,
            architecture=imsize48_default_architecture,
            decoder_distribution='gaussian_identity_variance',
            # decoder_distribution='laplace_identity_variance',
            # decoder_distribution='huber_identity_variance',
        ),
        decoder_activation=identity,
        representation_size=4,
        # default_path='/mnt/hdd/tung/workspace/rlkit/data/vae_adapt',
        default_path='/mnt/hdd/tung/workspace/rlkit/data/vae_adapt_new_data',
        beta=20,

        # TUNG: Change below
        vae_loss_opt=dict(
            loss_type='beta_vae_loss',
            # vae_loss='beta_vae_loss_rm_rec',
            # vae_loss='beta_vae_loss_stop_grad_dec',
            beta=20,
        ),
        da_loss_opt=dict(
            # loss_type='consistency_loss',
            loss_type='consistency_loss_w_cycle',
            # loss_type='consistency_loss_rm_dec_path',

            # distance=mse_loss,
            distance=l1_loss,
            # distance=huber_loss,

            alpha1=1.0,
            alpha2=1.0,
            alpha3=1.0,
            use_mu=True
        ),

        n_epochs=2000,
        step1=10000,  # Step to decay learning rate
        batch_size=50,
        lr=1e-3,

        init_tgt_by_src=False
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
    print('No. of random train data: Sim={}, Real={}'.format(len(rand_src_train),
                                                             len(rand_tgt_train)))
    print('No. of pair train data: Sim={}, Real={}'.format(len(pair_src_train),
                                                           len(pair_tgt_train)))

    ##########################
    #       Load models      #
    ##########################
    print('[INFO] Loading checkpoint...')
    data = torch.load(open(args.file, "rb"))
    env = data['evaluation/env']
    src_vae = env.vae

    tgt_vae = create_target_encoder(variant['vae_kwargs'],
                                    variant['representation_size'],
                                    variant['decoder_activation'])
    src_vae.to(ptu.device)
    tgt_vae.to(ptu.device)

    if variant['init_tgt_by_src']:
        tgt_vae.load_state_dict(src_vae.state_dict())

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

            loss = total_vae_loss + pair_loss
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
                  losses=losses, log_probs=log_probs, kles=kles, pair_losses=pair_losses)


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='path to the snapshot file')
parser.add_argument('--vae', type=str, default=None, help='path save checkpoint')
parser.add_argument('--exp', type=str, default=None, help='Another description for experiment name')
parser.add_argument('--gpu', action='store_false')
parser.add_argument('--seed', type=int, default=None)
user_args = parser.parse_args()


if __name__ == '__main__':
    main(user_args)
