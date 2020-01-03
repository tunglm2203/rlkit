import os
import numpy as np
import matplotlib.pyplot as plt


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def main():
    path = '/mnt/hdd/tung/workspace/rlkit/data/01-02-rlkit-pusher-rig-example/01-02-rlkit-pusher-rig-example_2020_01_02_13_14_05_0000--s-21885/vae_progress.csv'
    res = load_results(path)

    recon_mse = 'debug/MSE of reconstruction'
    random_mses = 'debug/MSE of random decoding'
    mse_improvement = 'debug/MSE improvement over random'

    train_loss = 'train/loss'
    train_logprob = 'train/Log Prob'
    train_kle = 'train/KL'
    test_loss = 'test/loss'
    test_logprob = 'test/Log Prob'
    test_kle = 'test/KL'

    mean_postfix = ' Mean'
    min_postfix = ' Min'
    max_postfix = ' Max'
    std_postfix = ' Std'

    # Mean variance
    plt.figure()
    plt.plot(res[recon_mse])
    plt.plot(res[random_mses + mean_postfix])
    plt.plot(res[mse_improvement + mean_postfix])
    plt.title('Debug (Mean)')
    plt.legend([recon_mse, random_mses + mean_postfix, mse_improvement + mean_postfix])

    # Max variance
    plt.figure()
    plt.plot(res[recon_mse])
    plt.plot(res[random_mses + max_postfix])
    plt.plot(res[mse_improvement + max_postfix])
    plt.title('Debug (Max)')
    plt.legend([recon_mse, random_mses + max_postfix, mse_improvement + max_postfix])

    # Min variance
    plt.figure()
    plt.plot(res[recon_mse])
    plt.plot(res[random_mses + min_postfix])
    plt.plot(res[mse_improvement + min_postfix])
    plt.title('Debug (Min)')
    plt.legend([recon_mse, random_mses + min_postfix, mse_improvement + min_postfix])

    # Std variance
    plt.figure()
    plt.plot(res[recon_mse])
    plt.plot(res[random_mses + std_postfix])
    plt.plot(res[mse_improvement + std_postfix])
    plt.title('Debug (Std)')
    plt.legend([recon_mse, random_mses + std_postfix, mse_improvement + std_postfix])

    # Loss variance
    plt.figure()
    plt.plot(res[train_loss])
    plt.plot(res[test_loss])
    plt.title('Debug (Loss)')
    plt.legend([train_loss, test_loss])

    # Log prob
    plt.figure()
    plt.plot(res[train_logprob])
    plt.plot(res[test_logprob])
    plt.title('Debug (Log Probability)')
    plt.legend([train_logprob, test_logprob])

    # KL
    plt.figure()
    plt.plot(res[train_kle])
    plt.plot(res[test_kle])
    plt.title('Debug (KL)')
    plt.legend([train_kle, test_kle])

    plt.show()


if __name__ == '__main__':
    main()

