import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


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


def main(args):
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
              (final_puck_dists[-1].mean(), final_puck_dists[-1].std(),
               final_hand_dists[-1].mean(), final_hand_dists[-1].std(),
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


parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, nargs='+')
parser.add_argument('--one', action='store_true')
parser.add_argument('--n_goals', type=int, default=30)
parser.add_argument('--error', type=str, nargs='+', default=[])
args_user = parser.parse_args()

if __name__ == '__main__':
    main(args_user)
