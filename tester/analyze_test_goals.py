import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from matplotlib.colors import LogNorm



def plot_data_distribution():
    # path = '/home/tung/workspace/rlkit/tester/rand_pair_sim_tgt.5000/'
    # path = '/home/tung/workspace/rlkit/tester/rand_img_sim_tgt_new.10000/'
    # path = '/home/tung/workspace/rlkit/tester/randlarger_pair_sim_tgt.500/'
    # path = '/home/tung/workspace/rlkit/tester/randlarger_pair_sim_tgt.1000/'
    # path = '/home/tung/workspace/rlkit/tester/randlarger_pair_sim_tgt.2000/'
    # path = '/home/tung/workspace/rlkit/tester/randlarger_pair_sim_tgt.3000/'
    # path = '/home/tung/workspace/rlkit/tester/randlarger_pair_real.1000/'
    path = '/home/tung/workspace/rlkit/tester/randlarge_pair_real_full.5000/'

    file = 'random_trajectories.npz'
    d = np.load(os.path.join(path, file))
    d = d['data']

    # ============= USER SCOPE =============
    ee_pos_key = 'ee_pos'
    ob_pos_key = 'obj_pos'
    margin_plot = 0.05

    n_eps, horizon = d.shape

    ee_pos = np.zeros((*d.shape, 2))
    ob_pos = np.zeros((*d.shape, 2))

    for i in range(n_eps):
        for j in range(horizon):
            ee_pos[i, j] = d[i, j][ee_pos_key]
            ob_pos[i, j] = d[i, j][ob_pos_key]

    ee_ob_pos = np.concatenate((ee_pos, ob_pos), axis=2)

    ee_pos_reshape = ee_pos.reshape(-1, 2)
    ob_pos_reshape = ob_pos.reshape(-1, 2)
    ee_ob_reshape = ee_ob_pos.reshape(-1, 4)

    # =================== DENSITY ESTIMATION ===================
    n_components = 1
    # clf1 = mixture.BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_distribution",
    #                                        n_components=3)
    # clf2 = mixture.BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_distribution",
    #                                        n_components=3)
    # clf3 = mixture.BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_distribution",
    #                                        n_components=3)

    clf1 = mixture.GaussianMixture(n_components=n_components)
    clf2 = mixture.GaussianMixture(n_components=n_components)
    clf3 = mixture.GaussianMixture(n_components=n_components)

    clf1.fit(ob_pos_reshape)
    pred1 = -clf1.score_samples(ob_pos_reshape)

    clf2.fit(ee_pos_reshape)
    pred2 = -clf2.score_samples(ee_pos_reshape)

    # clf3.fit(ee_ob_reshape)
    # pred3 = -clf3.score_samples(ee_ob_reshape)

    plt.subplot(221)
    plt.plot(pred1)
    plt.title('OB distribution')

    x = np.linspace(0.5 - margin_plot, 0.7 + margin_plot)
    y = np.linspace(-0.2 - margin_plot, 0.2 + margin_plot)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    plt.subplot(222)
    pred1.reshape(ob_pos_reshape.shape[0])
    Z1 = -clf1.score_samples(XX)
    Z1 = Z1.reshape(X.shape)
    CS1 = plt.contour(X, Y, Z1, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(-1, 5, 50))
    CB1 = plt.colorbar(CS1, shrink=0.8, extend='both')
    plt.scatter(ob_pos_reshape[:, 0], ob_pos_reshape[:, 1], .8)
    plt.axis('tight')
    plt.title('OB contour distribution')

    plt.subplot(223)
    plt.plot(pred2)
    plt.title('EE distribution')

    plt.subplot(224)
    pred2.reshape(ee_pos_reshape.shape[0])
    Z2 = -clf2.score_samples(XX)
    Z2 = Z2.reshape(X.shape)
    CS2 = plt.contour(X, Y, Z2, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(-1, 0, 50))
    CB2 = plt.colorbar(CS2, shrink=0.8, extend='both')
    plt.scatter(ee_pos_reshape[:, 0], ee_pos_reshape[:, 1], .8)
    plt.axis('tight')
    plt.title('EE contour distribution')

    # plt.figure()
    # plt.plot(pred3)
    # plt.title('EE-OB distribution')

    plt.show()


def display_pair_data():
    import glob
    import cv2
    from PIL import Image, ImageEnhance
    # rea_pair_path = '/home/tung/workspace/rlkit/tester/randlarger_pair_real.1000/'
    # sim_pair_path = '/home/tung/workspace/rlkit/tester/randlarger_pair_sim.1000/'
    # rea_pair_path = '/home/tung/workspace/rlkit/tester/rand_pair_real.1000/'
    # sim_pair_path = '/home/tung/workspace/rlkit/tester/rand_pair_sim.1000/'
    rea_pair_path = '/home/tung/workspace/rlkit/tester/randlarge_pair_real_full.5000/'
    sim_pair_path = '/home/tung/workspace/rlkit/tester/randlarge_pair_sim_full.5000/'
    zoom_size = 256
    im_size = 48

    rea_list = glob.glob(os.path.join(rea_pair_path, '*.npz'))
    rea_list.sort()
    if os.path.join(rea_pair_path, 'random_trajectories.npz') in rea_list:
        rea_list.remove(os.path.join(rea_pair_path, 'random_trajectories.npz'))

    sim_list = glob.glob(os.path.join(sim_pair_path, '*.npz'))
    sim_list.sort()

    n_imgs = len(sim_list)
    for i in range(n_imgs):
        print('Iter={}, name={}'.format(i, sim_list[i]))
        im_src = np.load(sim_list[i])['im'].reshape(3, im_size, im_size).transpose()[:, :, ::-1]
        im_src = cv2.resize(im_src, (zoom_size, zoom_size))
        im_tgt = np.load(rea_list[i])['im'].reshape(3, im_size, im_size).transpose()[:, :, ::-1]

        im_tgt = cv2.resize(im_tgt, (zoom_size, zoom_size))

        im = np.concatenate((im_src, im_tgt), axis=1)
        cv2.imshow('Pair images', im)
        cv2.waitKey(2)

        input('Enter...')


if __name__ == '__main__':
    plot_data_distribution()
    # display_pair_data()
