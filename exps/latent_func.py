# this script tests whether using latent functions (transformation of feature space to an embedding space first and then GP modelling in that space)
# is useful or not.

from utils import load_data, zero_mean_unit_variance, compute_rmse, entropy_from_cov
from models import GpytorchGPR
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import ipdb
from methods import greedy


def test_model(x_train, y_train, x_test, y_test, latent=None):
    gp = GpytorchGPR(latent=latent, lr=.05, max_iterations=500)
    gp.fit(x_train, y_train)
    pred = gp.predict(x_test)
    return gp, pred


# RESULT: with nonlinear latent, the log likelihood is better
# also, in the greedy setting, utility is more
if __name__ == '__main__':
    # features = ['plant_width_mean', 'plant_height_mean', 'dry_to_green_ratio_mean', 'yield']
    feature = 'plant_height_mean'
    file = 'data/' + feature + '_dataset.pkl'   
    nr, nc, X, Y = load_data(file)

    # only (row, range) 
    # X = X[:, :2]

    # split into training and test set
    n = len(Y)
    fraction = .1
    n_train = int(fraction*n)
    all_ind = np.random.permutation(range(n))
    train_ind = all_ind[:n_train]
    test_ind = all_ind[n_train:]

    x_train, y_train = X[train_ind], Y[train_ind]
    x_test, y_test = X[test_ind], Y[test_ind]
    mean, std = x_train[:,:2].mean(axis=0), x_train[:,:2].std(axis=0)
    x_train[:,:2] = zero_mean_unit_variance(x_train[:,:2], mean, std)
    x_test[:,:2] = zero_mean_unit_variance(x_test[:,:2], mean, std)
    
    gp, pred = test_model(x_train, y_train, x_test, y_test)
    gp_nl, pred_nl = test_model(x_train, y_train, x_test, y_test, latent='non_linear')

    rmse = compute_rmse(pred, y_test)
    rmse_nl = compute_rmse(pred_nl, y_test)

    # visualize distribution of mismatches
    # plt.figure(0)
    # sns.distplot(np.abs(pred - y_test))
    # plt.figure(1)
    # sns.distplot(np.abs(pred_nl - y_test))
    # plt.show()

    x_all = np.concatenate([x_train, x_test])
    sampled = np.full(len(x_all), False)
    sampled[train_ind] = True
    K = gp.cov_mat(x_all, x_all)
    num_samples = 30
    new_samples, cumm_utilies = greedy(x_all, sampled, K, num_samples)

    sampled_nl = np.full(len(x_all), False)
    sampled_nl[train_ind] = True
    K_nl = gp_nl.cov_mat(x_all, x_all)
    num_samples = 30
    new_samples_nl, cumm_utilies_nl = greedy(x_all, sampled_nl, K_nl, num_samples)

    ipdb.set_trace()