import os
import glob
import datetime
import numpy as np
import dateutil.tz

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
            env_state = src.wrapped_env.get_env_state()
            src.wrapped_env.set_to_goal(src.wrapped_env.get_goal())
            env_state_ = src.wrapped_env.get_env_state()
            target.wrapped_env.set_env_state(env_state_)
            src.wrapped_env.set_env_state(env_state)
        else:
            """ New method to set state, only possible in Mujoco Environment
            """
            # Get state
            env_state = src.wrapped_env.get_env_state()
            # Set state
            target.wrapped_env.set_env_state(env_state)

            # """ Old method to set state, possible in Real and Gazebo
            # """
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
        # Get coordinate real
        ee_pos = src.wrapped_env._get_endeffector_pose()
        obj_pos = src.wrapped_env.get_obj_pos_in_gazebo('cylinder')
        # Make the coordinate of Mujoco
        target._goal_xyxy = np.zeros(4)
        target._goal_xyxy[:2] = np.array(ee_pos[:2])  # EE
        target._goal_xyxy[2:] = np.array(obj_pos[:2])  # OBJ
        target.set_goal_xyxy(target._goal_xyxy)
        # target.reset_mocap_welds()
        # target._get_obs()
        target.wrapped_env.set_to_goal(target.wrapped_env.get_goal())
    else:
        print('[AIM-ERROR] Only support ImageEnv, this is {}'.format(type(src)))
        exit()
    pass


def set_env_state_sim2real():
    pass
