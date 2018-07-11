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


def mi_change(x, A, A_bar, kernel, noise_var):
    """
    :param x:
    :param A:
    :param A_bar:
    :param kernel: Kernel
    :param noise_var: measurement noise variance
    :return: H(x|A) - H(x|A_bar)
    """
    e1 = conditional_entropy(x, A, kernel, noise_var)
    e2 = conditional_entropy(x, A_bar, kernel, noise_var)
    info = e1 - e2
    return info


def conditional_entropy(x, A, kernel, noise_var):
    """
    :param x:
    :param A:
    :param kernel: Kernel
    :param noise_var: variance due to measurement noise
    :return: H(x|A)
    """
    x_ = x.reshape(-1, A.shape[-1])
    sigma_AA = kernel(A, A) + noise_var * np.eye(A.shape[0])
    sigma_xA = kernel(x_, A)
    cov = kernel(x_, x_) - np.dot(np.dot(sigma_xA, np.linalg.inv(sigma_AA)), sigma_xA.T)
    d = cov.shape[0]
    ent = d*CONST + .5*np.log(np.linalg.det(cov))
    return ent
