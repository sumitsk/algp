import numpy as np
import ipdb

CONST = .5*np.log(2*np.pi*np.exp(1))


def generate_gaussian_data(num_rows, num_cols, k=5, min_var=10, max_var=100, algo='sum'):
    """
    :param num_rows: number of rows
    :param num_cols: number of columns
    :param k: number of gaussian component
    :param min_var: minimum variance
    :param max_var: maximum variance
    :param algo: sum / max of mixture of Gaussians
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
        if algo == 'max':
            y = np.maximum(y, tmp)
        elif algo == 'sum':
            y += tmp

    return grid, y

# @deprecated
# def conditional_entropy(x, model):
#     """
#     compute entropy of set x conditioned on the training set of GP model
#     :param x: test locations
#     :param model: GP model
#     :return: H(x| model.X_train_)
#     """
#
#     dim = model.X_train_.shape[-1]
#     x_ = x.reshape(-1, dim)
#     mu, cov = model.predict(x_, return_cov=True)
#     d = cov.shape[0]
#     ent = d*CONST + .5*np.log(np.linalg.det(cov))
#     return ent

# @deprecated
# def mutual_information(x, model):
#     """
#     compute mutual information between set X and training set of GP model
#     :param x: test locations
#     :param model: GP model
#     :return: MI(x, model.X_train_)
#     """
#
#     dim = model.X_train_.shape[-1]
#     x_ = x.reshape(-1, dim)
#     # todo: should noise be added too ?
#     cov = model.kernel_(x_, x_)  # + model.alpha * np.eye(x_.shape[0])
#     d = cov.shape[0]
#     ent = d*CONST + .5*np.log(np.linalg.det(cov))
#     cond_ent = conditional_entropy(x, model)
#     mi = ent - cond_ent
#     return mi


def mi_change(x, a, a_bar, kernel, x_noise_var=None, a_noise_var=None, a_bar_noise_var=None):
    e1 = conditional_entropy(x, a, kernel, x_noise_var, a_noise_var)
    e2 = conditional_entropy(x, a_bar, kernel, x_noise_var, a_bar_noise_var)
    info = e1 - e2
    return info


def process_noise_var(dim, noise_var):
    if noise_var is None:
        noise_var_ = 0
    elif isinstance(noise_var, int) or isinstance(noise_var, float):
        noise_var_ = noise_var * np.eye(dim)
    elif isinstance(noise_var, list) or isinstance(noise_var, np.ndarray):
        assert len(noise_var) == dim, 'Size mismatch!!'
        noise_var_ = np.diag(noise_var)
    else:
        raise NotImplementedError
    return noise_var_


def entropy(x, kernel, x_noise_var):
    x_ = np.copy(x)
    if x.ndim == 1:
        x_ = x.reshape(-1, len(x))

    x_noise_var_ = process_noise_var(x_.shape[0], x_noise_var)
    cov = kernel(x_, x_) + x_noise_var_
    d = cov.shape[0]
    ent = d*CONST + .5*np.log(np.linalg.det(cov))
    return ent


def conditional_entropy(x, a, kernel, x_noise_var, a_noise_var, sigma_aa_inv=None):
    assert a.ndim == 2, 'Matrix A must be 2-dimensional!'
    if a.shape[0] == 0:
        return entropy(x, kernel, x_noise_var)

    x_ = x.reshape(-1, a.shape[-1])
    x_noise_var_ = process_noise_var(x_.shape[0], x_noise_var)
    a_noise_var_ = process_noise_var(a.shape[0], a_noise_var)

    if sigma_aa_inv is None:
        sigma_aa_inv = np.linalg.inv(kernel(a, a) + a_noise_var_)
    sigma_xa = kernel(x_, a)
    sigma_xx = kernel(x_, x_) + x_noise_var_
    cov = sigma_xx - np.dot(np.dot(sigma_xa, sigma_aa_inv), sigma_xa.T)
    d = cov.shape[0]
    ent = d*CONST + .5*np.log(np.linalg.det(cov))
    return ent


def is_valid_cell(cell, grid_shape):
    if 0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]:
        return True
    return False

