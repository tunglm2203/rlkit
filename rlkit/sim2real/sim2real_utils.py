import os
import glob
import json
import random
import datetime
import numpy as np
import dateutil.tz
from collections import OrderedDict
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
from torch.nn import functional as F

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.sim2real.goal_test import set_of_goals
from rlkit.samplers.rollout_functions import multitask_rollout_sim2real
from rlkit.core import logger
from rlkit.core.logging import MyEncoder
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.vae.conv_vae import (ConvVAE,)
from rlkit.sim2real.sim2real_losses import *


def create_target_encoder(vae_kwargs, representation_size, decoder_activation):
    conv_vae = ConvVAE(
        representation_size,
        decoder_output_activation=decoder_activation,
        **vae_kwargs
    )
    return conv_vae


def load_all_required_data(path='', rand_src_dir=None, rand_tgt_dir=None,
                           pair_src_dir=None, pair_tgt_dir=None,
                           file_format='*.npz',
                           test_ratio=0.1, seed=0,
                           merge_rand_pair_tgt=False):
    rand_sim_train, rand_real_train, rand_sim_eval, rand_real_eval = None, None, None, None
    if rand_src_dir is not None:
        rand_data_sim = load_data(os.path.join(path, rand_src_dir), file_format)
    if rand_tgt_dir is not None:
        rand_data_real = load_data(os.path.join(path, rand_tgt_dir), file_format)
    pair_data_sim = load_data(os.path.join(path, pair_src_dir), file_format)
    pair_data_real = load_data(os.path.join(path, pair_tgt_dir), file_format)

    if merge_rand_pair_tgt and rand_tgt_dir is not None:
        rand_data_real = np.concatenate((rand_data_real, pair_data_real), axis=0)
    if merge_rand_pair_tgt and rand_src_dir is not None:
        rand_data_sim = np.concatenate((rand_data_sim, pair_data_sim), axis=0)

    if rand_src_dir is not None and rand_tgt_dir is not None:
        assert len(rand_data_sim) == len(rand_data_real), \
            "[ERROR] Number of random sim & real data not equal"
    assert len(pair_data_sim) == len(pair_data_real), \
        "[ERROR] Number of pair sim & real data not equal"
    if rand_src_dir is not None and rand_tgt_dir is not None:
        n_rand_trains = int(len(rand_data_sim) * (1 - test_ratio))
    n_pair_trains = int(len(pair_data_sim) * (1 - test_ratio))

    st0 = np.random.get_state()     # Get current state (state 0)
    np.random.seed(seed)
    if rand_src_dir is not None and rand_tgt_dir is not None:
        idxes_rand = np.random.permutation(len(rand_data_sim))
    idxes_pair = np.random.permutation(len(pair_data_sim))
    np.random.set_state(st0)        # Return to state 0

    if rand_src_dir is not None and rand_tgt_dir is not None:
        rand_sim_train = rand_data_sim[idxes_rand[:n_rand_trains]]
        rand_real_train = rand_data_real[idxes_rand[:n_rand_trains]]
        rand_sim_eval = rand_data_sim[idxes_rand[n_rand_trains:]]
        rand_real_eval = rand_data_real[idxes_rand[n_rand_trains:]]

    pair_sim_train = pair_data_sim[idxes_pair[:n_pair_trains]]
    pair_sim_eval = pair_data_sim[idxes_pair[n_pair_trains:]]
    pair_real_train = pair_data_real[idxes_pair[:n_pair_trains]]
    pair_real_eval = pair_data_real[idxes_pair[n_pair_trains:]]

    return rand_sim_train, rand_real_train, rand_sim_eval, rand_real_eval, \
           pair_sim_train, pair_sim_eval, pair_real_train, pair_real_eval


def load_data(path='', file_format='*.npz'):
    images = []
    if os.path.exists(path):
        filenames = glob.glob(os.path.join(path, file_format))
        filenames.sort()
        for i in range(len(filenames)):
            im_sim = np.load(filenames[i])
            images.append(im_sim['im'])
    else:
        print('[ERROR] Cannot find data')
    return np.array(images)


def setup_logger_path(args, path, seed):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
    prefix_dir = 'vae-exp-date-' + timestamp[:10]
    sub_dir = 'vae-exp-time-' + timestamp[11:19].replace(':', '-')
    if args.exp is not None:
        ckpt_path = os.path.join(path, prefix_dir, args.exp + '-' + sub_dir + '-s' + str(seed))
    else:
        ckpt_path = os.path.join(path, prefix_dir, 'None-' + sub_dir + '-s' + str(seed))

    if not os.path.exists(os.path.join(path, prefix_dir)):
        os.mkdir(os.path.join(path, prefix_dir))
    os.mkdir(ckpt_path)
    return ckpt_path


def convert_vec2img_3_48_48(vec):
    """
    Converting image in vector to matrix.
    :param vec: shape of N x 6912
    :return: mat: shape of N x 48 x 48 x 3
    """
    if len(vec.shape) == 1:
        assert vec.shape[0] == 6912, "Single, shape of image is not 48x48x3"
        mat = vec.reshape(3, 48, 48).transpose()[:, :, ::-1]    # inverse color channel for cv2
    elif len(vec.shape) == 2:
        assert vec.shape[1] == 6912, "Batch, shape of image is not 48x48x3"
        raise NotImplementedError()
    return mat


def set_env_state_sim2sim(src, target, set_goal=False):
    """
    Function to set state from sim (Mujoco) to sim (Mujoco)
    :param src: Env want to get state
    :param target: Env want to set state from src's state
    :param set_goal: This call for set goal or not
    :return:
    """
    from multiworld.core.image_env import ImageEnv
    if isinstance(src, ImageEnv) and isinstance(target, ImageEnv):
        if set_goal:
            """ New method to set state, only possible in Mujoco Environment
            """
            env_state = src.wrapped_env.get_env_state()
            src.wrapped_env.set_to_goal(src.wrapped_env.get_goal())
            env_state_ = src.wrapped_env.get_env_state()
            target.wrapped_env.set_env_state(env_state_)
            src.wrapped_env.set_env_state(env_state)
            """ Old method to set state, possible in Real and Gazebo
            """
            # goals = src.wrapped_env.get_goal()
            # obj_pos = goals['state_desired_goal'][2:]
            # ee_pos = goals['state_desired_goal'][:2]
            # target.wrapped_env._goal_xyxy = np.zeros(4)
            # target.wrapped_env._goal_xyxy[:2] = np.array(ee_pos[:2])  # EE
            # target.wrapped_env._goal_xyxy[2:] = np.array(obj_pos[:2])  # OBJ
            # target.set_goal_xyxy(target._goal_xyxy)
            # target.wrapped_env.set_to_goal(target.wrapped_env.get_goal())
        else:
            """ New method to set state, only possible in Mujoco Environment
            """
            # Get state
            env_state = src.wrapped_env.get_env_state()
            # Set state
            target.wrapped_env.set_env_state(env_state)

            """ Old method to set state, possible in Real and Gazebo
            """
            # # Get coordinate real
            # ee_pos = src.wrapped_env.get_endeff_pos()
            # obj_pos = src.wrapped_env.get_puck_pos()
            # # Make the coordinate of Mujoco
            # target.wrapped_env._goal_xyxy = np.zeros(4)
            # target.wrapped_env._goal_xyxy[:2] = np.array(ee_pos[:2])  # EE
            # target.wrapped_env._goal_xyxy[2:] = np.array(obj_pos[:2])  # OBJ
            # target.set_goal_xyxy(target._goal_xyxy)
            # target.wrapped_env.set_to_goal(target.wrapped_env.get_goal())
    else:
        print('[AIM-ERROR] Only support ImageEnv, this is {}'.format(type(src)))
        exit()
    return 0


def set_env_state_real2real():
    pass


def set_env_state_real2sim(src, target, set_goal=False):
    """
    Function to set state from real (Gazebo) to sim (Mujoco)
    :param src: Env want to get state
    :param target: Env want to set state from src's state
    :param set_goal: This call for set goal or not
    :return:
    """
    from multiworld.core.image_env import ImageEnv
    if isinstance(src, ImageEnv) and isinstance(target, ImageEnv):
        if set_goal:
            goals = src.wrapped_env.get_goal()
            ee_pos = goals['state_desired_goal'][2:]
            obj_pos = goals['state_desired_goal'][:2]
        else:
            # Get coordinate real
            ee_pos = src.wrapped_env._get_endeffector_pose()
            obj_pos = src.wrapped_env.get_obj_pos_in_gazebo('cylinder')

        # Make the coordinate of Mujoco
        target.wrapped_env._goal_xyxy = np.zeros(4)
        target.wrapped_env._goal_xyxy[:2] = np.array(ee_pos[:2])  # EE
        target.wrapped_env._goal_xyxy[2:] = np.array(obj_pos[:2])  # OBJ
        target.wrapped_env.set_goal_xyxy(target.wrapped_env._goal_xyxy)
        # target.reset_mocap_welds()
        # target._get_obs()
        target.wrapped_env.set_to_goal(target.wrapped_env.get_goal())
    else:
        print('[AIM-ERROR] Only support ImageEnv, this is {}'.format(type(src)))
        exit()


def set_env_state_sim2real():
    pass


def setup_logger(args, variant):
    ckpt_path = setup_logger_path(args=args, path=variant['default_path'], seed=variant['seed'])
    tensor_board = SummaryWriter(ckpt_path)
    with open(os.path.join(ckpt_path, 'log_params.json'), "w") as f:
        json.dump(variant, f, indent=2, sort_keys=True, cls=MyEncoder)
    eval_statistics = OrderedDict()
    logger.set_snapshot_dir(ckpt_path)
    logger.add_tabular_output(
        'vae_progress.csv', relative_to_snapshot_dir=True
    )

    return eval_statistics, tensor_board, ckpt_path


def set_seed(seed):
    """
    Set the seed for all the possible random number generators.

    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_batch(data, batch_idx, idxes, batch_size):
    _n_batchs = len(idxes) // batch_size
    if batch_idx >= _n_batchs:
        batch_idx = batch_idx % _n_batchs
    batch_data = ptu.from_numpy(data[idxes[batch_idx * batch_size:(batch_idx + 1) * batch_size]])
    return batch_data


def mse_loss(inputs, targets, imlength=None):
    if imlength is not None:
        inputs = inputs.narrow(start=0, length=imlength, dim=1).contiguous().view(-1, imlength)
    loss = F.mse_loss(inputs, targets, reduction='elementwise_mean')
    return loss


def huber_loss(inputs, targets, imlength=None):
    if imlength is not None:
        inputs = inputs.narrow(start=0, length=imlength, dim=1).contiguous().view(-1, imlength)
    loss = F.smooth_l1_loss(targets, inputs, reduction='elementwise_mean')
    return loss


def l1_loss(inputs, targets, imlength=None):
    if imlength is not None:
        inputs = inputs.narrow(start=0, length=imlength, dim=1).contiguous().view(-1, imlength)
    loss = F.l1_loss(inputs, targets, reduction='elementwise_mean')
    return loss


def pairwise_loss_schedule(value, epoch):
    if epoch < 500:
        return 0.0
    else:
        return value


def vae_da_loss_schedule_v1(epoch, step, vae_loss_opt, da_loss_opt):
    """
    Used for VAE & consistency loss
    """
    _vae_loss_opt = vae_loss_opt.copy()
    _da_loss_opt = da_loss_opt.copy()
    if epoch < step:
        _vae_loss_opt['alpha0'] = 1.0 * vae_loss_opt['alpha0']
        _da_loss_opt['alpha1'] = 0.0 * da_loss_opt['alpha1']
    else:
        _vae_loss_opt['alpha0'] = 1.0 * vae_loss_opt['alpha0']
        _da_loss_opt['alpha1'] = 1.0 * da_loss_opt['alpha1']
    return _vae_loss_opt, _da_loss_opt


def vae_da_loss_schedule_v2(epoch, step, vae_loss_opt, da_loss_opt):
    """
    Used for VAE & consistency loss
    """
    _vae_loss_opt = vae_loss_opt.copy()
    _da_loss_opt = da_loss_opt.copy()
    if epoch < step:
        _vae_loss_opt['alpha0'] = 1.0 * vae_loss_opt['alpha0']
        _da_loss_opt['alpha1'] = 0.0 * da_loss_opt['alpha1']
    else:
        _vae_loss_opt['alpha0'] = 0.0 * vae_loss_opt['alpha0']
        _da_loss_opt['alpha1'] = 1.0 * da_loss_opt['alpha1']
    return _vae_loss_opt, _da_loss_opt


def vae_da_loss_schedule_v3(epoch, step, vae_loss_opt, da_loss_opt):
    """
    Used for VAE & consistency w/ cycle loss
    """
    _vae_loss_opt = vae_loss_opt.copy()
    _da_loss_opt = da_loss_opt.copy()
    if epoch < step:
        _vae_loss_opt['alpha0'] = 1.0 * vae_loss_opt['alpha0']
        _da_loss_opt['alpha1'] = 0.0 * da_loss_opt['alpha1']
        _da_loss_opt['alpha2'] = 0.0 * da_loss_opt['alpha2']
        _da_loss_opt['alpha3'] = 0.0 * da_loss_opt['alpha3']
    else:
        _vae_loss_opt['alpha0'] = 1.0 * vae_loss_opt['alpha0']
        _da_loss_opt['alpha1'] = 1.0 * da_loss_opt['alpha1']
        _da_loss_opt['alpha2'] = 1.0 * da_loss_opt['alpha2']
        _da_loss_opt['alpha3'] = 1.0 * da_loss_opt['alpha3']
    return _vae_loss_opt, _da_loss_opt


def vae_da_loss_schedule_v4(epoch, step, vae_loss_opt, da_loss_opt):
    """
    Used for VAE & consistency w/ cycle loss
    """
    _vae_loss_opt = vae_loss_opt.copy()
    _da_loss_opt = da_loss_opt.copy()
    if epoch < step:
        _vae_loss_opt['alpha0'] = 1.0 * vae_loss_opt['alpha0']
        _da_loss_opt['alpha1'] = 0.0 * da_loss_opt['alpha1']
        _da_loss_opt['alpha2'] = 0.0 * da_loss_opt['alpha2']
        _da_loss_opt['alpha3'] = 0.0 * da_loss_opt['alpha3']
    else:
        _vae_loss_opt['alpha0'] = 0.0 * vae_loss_opt['alpha0']
        _da_loss_opt['alpha1'] = 1.0 * da_loss_opt['alpha1']
        _da_loss_opt['alpha2'] = 1.0 * da_loss_opt['alpha2']
        _da_loss_opt['alpha3'] = 1.0 * da_loss_opt['alpha3']
    return _vae_loss_opt, _da_loss_opt


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


def create_vae_wrapped_env(env_name, vae, imsize=48, init_camera=None):
    import gym
    from multiworld.core.image_env import ImageEnv
    import multiworld
    from rlkit.envs.vae_wrapper import VAEWrappedEnv
    multiworld.register_all_envs()

    env = gym.make(env_name)
    image_env = ImageEnv(env,
                         imsize=imsize,
                         normalize=True,
                         transpose=True,
                         init_camera=init_camera,
                         )
    # TUNG: ====== Hard code: clone from configuration of data_ckpt['evaluation/env']
    render = False
    reward_params = dict(
        type='latent_distance',
    )

    vae_env = VAEWrappedEnv(
        image_env,
        vae,
        imsize=image_env.imsize,
        decode_goals=render,
        render_goals=render,
        render_rollouts=render,
        reward_params=reward_params,
    )

    # TUNG: Using a goal sampled from environment
    vae_env.wrapped_env.wrapped_env.randomize_goals = False  # Setup Mujoco env
    vae_env._goal_sampling_mode = 'reset_of_env'             # Setup VAE env
    return vae_env


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
        tb.add_scalar('eval/puck_distance (std)', ob_dist_std, epoch)
        tb.add_scalar('eval/hand_distance (mean)', ee_dist_mean, epoch)
        tb.add_scalar('eval/hand_distance (std)', ee_dist_std, epoch)

    for key, value in stats.items():
        tb.add_scalar(key, value, epoch)