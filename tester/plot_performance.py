import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from rlkit.sim2real.sim2real_utils import convert_vec2img_3_48_48


def plot_mean_std(data):
    """
    Plot the data with mean and std, std is shaded region
    :param data: shape of N x m, N is number of value in x-axis, m is number of variances of a
                 value of x-axis.
    """
    N, m = data.shape
    x = np.arange(N)
    y_mean = np.nanmean(data[:, :m], axis=1)
    plt.plot(x, y_mean, '-o', markersize=3)
    plt.fill_between(x,
                     np.nanmean(data[:, :m], axis=1) + np.nanstd(data[:, :m], axis=1),
                     np.nanmean(data[:, :m], axis=1) - np.nanstd(data[:, :m], axis=1),
                     alpha=0.25)
    # label indicating episode number
    for i in range(N):
        plt.text(x[i] + .03, y_mean[i], str(i), fontsize=8)


def plot_performance(args):
    error = []
    if len(args.error) > 0:
        for i in range(len(args.error)):
            error.append(int(args.error[i]))

    collect_data, legends = [], []
    final_puck_dists, final_hand_dists = [], []

    # ============= REGION TO COMPUTE STATISTIC TO PRINT =============
    print(' ======================== Average results ========================')
    # Scan list of directories
    for i in range(len(args.dir)):
        if args.dir[i].split('/')[-1] != '':
            legends.append(args.dir[i].split('/')[-1])
        elif args.dir[i].split('/')[-2] != '':
            legends.append(args.dir[i].split('/')[-2])
        else:
            print('Cannot find legend of {}'.format(args.dir[i]))

        # Read the stored result
        collect_data.append(np.load(os.path.join(args.dir[i], 'test_policy.npz')))
        n_goals = len(collect_data[i]['final_puck_distance'])

        # Pre-process result
        if args.one:
            idxes_for_correct_exp = []
            # This loop to get the index of correct experiments
            for g_id in range(n_goals):
                if g_id not in error:
                    idxes_for_correct_exp.append(g_id)
            final_puck_dists.append(collect_data[i]['final_puck_distance'][idxes_for_correct_exp])
            final_hand_dists.append(collect_data[i]['final_hand_distance'][idxes_for_correct_exp])
        else:
            final_puck_dists.append(collect_data[i]['final_puck_distance'])
            final_hand_dists.append(collect_data[i]['final_hand_distance'])

        print("Result: puck_distance=%.4f/%.4f, hand_distance=%.4f/%.4f, exp=%s" %
              (final_puck_dists[-1].mean(), final_puck_dists[-1].mean(axis=0).std(),
               final_hand_dists[-1].mean(), final_hand_dists[-1].mean(axis=0).std(),
               legends[i]))
    print(' =================================================================')

    # ============= REGION TO COMPUTE STATISTIC TO PLOT =============
    # Puck distance (Object)
    plt.figure()
    for i in range(len(args.dir)):
        plot_mean_std(final_puck_dists[i])
    plt.legend(legends)
    plt.title('final_puck_distance')
    # plt.ylim((0.0, 0.2))

    # Hand distance (End effector)
    plt.figure()
    for i in range(len(args.dir)):
        plot_mean_std(final_hand_dists[i])
    plt.legend(legends)
    plt.title('final_hand_distance')
    # plt.ylim((0.0, 0.2))

    plt.show()


def replay_epside(args):
    path = os.path.join(args.dir[0], 'episodes.npz')
    episode_start = args.start

    large_size = (128, 128)
    key_im_obs_ori = 'image_observation'
    key_im_obs_rec = 'image_observation_rec'
    key_im_goal_orig = 'image_desired_goal_orig'
    key_im_goal_rec = 'image_desired_goal'
    im_obs_mujoco_key = 'image_observation_mujoco'
    im_goal_mujoco_key = 'image_desired_goal_mujoco'

    data = np.load(path)
    data = data['episode']

    N, m = data.shape
    horizon = data[0][0]['observations'].shape[0]

    gen_mujoco_im = False
    if im_obs_mujoco_key in data[0][0]['next_observations'][0].keys() and \
            im_goal_mujoco_key in data[0][0]['next_observations'][0].keys():
        gen_mujoco_im = True

    if args.mujoco:
        test_policy = np.load(os.path.join(args.dir[0], 'test_policy.npz'))
        final_puck_dists = test_policy['final_puck_distance']
        final_hand_dists = test_policy['final_hand_distance']

        final_puck_dists_mujoco = np.zeros_like(final_puck_dists)
        final_hand_dists_mujoco = np.zeros_like(final_hand_dists)

        for i in range(N):
            for j in range(m):
                final_puck_dists_mujoco[i, j] = data[i, j]['env_infos'][-1]['puck_distance_mujoco']
                final_hand_dists_mujoco[i, j] = data[i, j]['env_infos'][-1]['hand_distance_mujoco']

        plt.figure()
        plot_mean_std(final_puck_dists)
        plot_mean_std(final_puck_dists_mujoco)
        plt.legend(['real', 'sim'])
        plt.title('final_puck_distance')

        plt.figure()
        plot_mean_std(final_hand_dists)
        plot_mean_std(final_hand_dists_mujoco)
        plt.legend(['real', 'sim'])
        plt.title('final_hand_distance')
        plt.show(block=False)

    for eps in range(episode_start, N):
        for t in range(m):
            for h in range(horizon):
                im_obs_orig = data[eps][t]['next_observations'][h][key_im_obs_ori]
                im_obs_rec = data[eps][t]['next_observations'][h][key_im_obs_rec]
                im_goal_orig = data[eps][t]['next_observations'][h][key_im_goal_orig]
                im_goal_rec = data[eps][t]['next_observations'][h][key_im_goal_rec]
                if gen_mujoco_im:
                    im_obs_orig_mujoco = data[eps][t]['next_observations'][h][im_obs_mujoco_key]
                    im_goal_orig_mujoco = data[eps][t]['next_observations'][h][im_goal_mujoco_key]
                    im_obs_orig_mujoco = cv2.resize(convert_vec2img_3_48_48(im_obs_orig_mujoco), large_size)
                    im_goal_orig_mujoco = cv2.resize(convert_vec2img_3_48_48(im_goal_orig_mujoco), large_size)

                im_obs_orig = cv2.resize(convert_vec2img_3_48_48(im_obs_orig), large_size)
                im_obs_rec = cv2.resize(convert_vec2img_3_48_48(im_obs_rec), large_size)
                # im_obs_rec = cv2.resize(im_obs_rec, large_size)
                im_goal_orig = cv2.resize(convert_vec2img_3_48_48(im_goal_orig), large_size)
                im_goal_rec = cv2.resize(convert_vec2img_3_48_48(im_goal_rec), large_size)

                if gen_mujoco_im:
                    im_0 = np.concatenate((im_obs_orig, im_goal_orig), axis=1)
                    im_1 = np.concatenate((im_obs_orig_mujoco, im_obs_rec), axis=1)
                    im_2 = np.concatenate((im_goal_orig_mujoco, im_goal_rec), axis=1)
                    im = np.concatenate((im_0, im_1, im_2), axis=0)
                else:
                    im_1 = np.concatenate((im_obs_orig, im_obs_rec), axis=1)
                    im_2 = np.concatenate((im_goal_orig, im_goal_rec), axis=1)
                    im = np.concatenate((im_1, im_2), axis=0)
                cv2.imshow('Observation/Goal', im)
                cv2.waitKey(1)
                print('Episode/test/step: {}/{}/{}'.format(eps, t, h))
                input('Press Enter...')
        # cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, nargs='+')
parser.add_argument('--one', action='store_true')
parser.add_argument('--n_goals', type=int, default=30)
parser.add_argument('--error', type=str, nargs='+', default=[])
parser.add_argument('--mujoco', action='store_true')

parser.add_argument('--replay', action='store_true')
parser.add_argument('--start', type=int, default=0)
args_user = parser.parse_args()

if __name__ == '__main__':
    if args_user.replay:
        replay_epside(args_user)
    else:
        plot_performance(args_user)
