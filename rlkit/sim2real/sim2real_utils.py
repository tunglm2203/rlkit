import os
import glob
import json
import random
import datetime
import numpy as np
import dateutil.tz
from collections import OrderedDict
from tensorboardX import SummaryWriter

import torch
from torch.nn import functional as F

from rlkit.core import logger
from rlkit.core.logging import MyEncoder
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.vae.conv_vae import (ConvVAE,)


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
                           test_ratio=0.1, seed=0):
    rand_sim_train, rand_real_train, rand_sim_eval, rand_real_eval = None, None, None, None
    if rand_src_dir is not None:
        rand_data_sim = load_data(os.path.join(path, rand_src_dir), file_format)
    if rand_tgt_dir is not None:
        rand_data_real = load_data(os.path.join(path, rand_tgt_dir), file_format)
    pair_data_sim = load_data(os.path.join(path, pair_src_dir), file_format)
    pair_data_real = load_data(os.path.join(path, pair_tgt_dir), file_format)

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
