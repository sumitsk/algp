import numpy as np
import ipdb

CONST = .5*np.log(2*np.pi*np.exp(1))


def generate_gaussian_data(num_rows, num_cols, k=5, min_var=100, max_var=1000, type='sum'):
    """
    :param num_rows: number of rows
    :param num_cols: number of columns
    :param k: number of gaussian component
    :param min_var: minimum variance
    :param max_var: maximum variance
    :param type: sum / max of mixture of gaussians
    :return:
    """
    x, y = np.meshgrid(np.arange(num_rows), np.arange(num_cols))
    grid = np.vstack([y.flatten(), x.flatten()]).transpose()

    means_x = np.random.uniform(0, num_rows, size=k)
    means_y = np.random.uniform(0, num_cols, size=k)
    means = np.vstack([means_x, means_y]).transpose()
    variances = np.random.uniform(min_var, max_var, size=k)

    y = np.zeros(num_rows * num_cols)
    for i in range(k):
        dist_sq = np.sum(np.square(grid - means[i].reshape(1, -1)), axis=1)
        tmp = np.exp(-dist_sq / variances[i])
        if type == 'max':
            y = np.maximum(y, tmp)
        elif type == 'sum':
            y += tmp

    return grid, y


def conditional_entropy(x, model):
    """
    compute entropy of set x conditioned on the training set of GP model
    :param x: test locations
    :param model: GP model
    :return: H(x| model.X_train_)
    """

    dim = model.X_train_.shape[-1]
    x_ = x.reshape(-1, dim)
    mu, cov = model.predict(x_, return_cov=True)
    d = cov.shape[0]
    ent = d*CONST + .5*np.log(np.linalg.det(cov))
    return ent


def mutual_information(x, model, noise_std=.01):
    """
    compute mutual information between set X and training set of GP model
    :param x: test locations
    :param model: GP model
    :return: MI(x, model.X_train_)
    """

    dim = model.X_train_.shape[-1]
    x_ = x.reshape(-1, dim)
    # todo: should noise be added too ?
    cov = model.kernel_(x_, x_) + noise_std**2 * np.eye(x_.shape[0])
    d = cov.shape[0]
    ent = d*CONST + .5*np.log(np.linalg.det(cov))
    cond_ent = conditional_entropy(x, model)
    mi = ent - cond_ent
    return mi


# @deprecated
# def my_conditional_entropy(x, model, noise_std=.01):
#     dim = model.X_train_.shape[-1]
#     x_ = x.reshape(-1, dim)
#     kernel = model.kernel_
#     A = model.X_train_
#     sigma_AA = kernel(A, A) + noise_std**2 * np.eye(A.shape[0])
#     sigma_xA = kernel(x_, A)
#     cov = kernel(x_, x_) - np.dot(np.dot(sigma_xA, np.linalg.inv(sigma_AA)), sigma_xA.T)
#     d = cov.shape[0]
#     ent = d*CONST + .5*np.log(np.linalg.det(cov))
#     return ent
