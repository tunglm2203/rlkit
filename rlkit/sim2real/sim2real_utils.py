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
    rand_data_sim = load_data(os.path.join(path, rand_src_dir), file_format)
    rand_data_real = load_data(os.path.join(path, rand_tgt_dir), file_format)
    pair_data_sim = load_data(os.path.join(path, pair_src_dir), file_format)
    pair_data_real = load_data(os.path.join(path, pair_tgt_dir), file_format)
    assert len(rand_data_sim) == len(rand_data_real), \
        "[ERROR] Number of random sim & real data not equal"
    assert len(pair_data_sim) == len(pair_data_real), \
        "[ERROR] Number of pair sim & real data not equal"
    n_rand_trains = int(len(rand_data_sim) * (1 - test_ratio))
    n_pair_trains = int(len(pair_data_sim) * (1 - test_ratio))

    st0 = np.random.get_state()     # Get current state (state 0)
    np.random.seed(seed)
    idxes_rand = np.random.permutation(len(rand_data_sim))
    idxes_pair = np.random.permutation(len(pair_data_sim))
    np.random.set_state(st0)        # Return to state 0

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


def setup_logger_path(args, path):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
    prefix_dir = 'vae-exp-date-' + timestamp[:10]
    sub_dir = 'vae-exp-time-' + timestamp[11:19].replace(':', '-')
    if args.exp is not None:
        ckpt_path = os.path.join(path, prefix_dir, sub_dir + '-' + args.exp)
    else:
        ckpt_path = os.path.join(path, prefix_dir, sub_dir)

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
