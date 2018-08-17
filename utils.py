import numpy as np
import ipdb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import pickle 


CONST = .5*np.log(2*np.pi*np.exp(1))


def to_torch(arr):
    if arr is None:
        return None
    if arr.__class__.__module__ == 'torch':
        return arr
    if arr.__class__.__module__ == 'numpy':
        return torch.FloatTensor(arr)
    return arr


def to_numpy(x):
    if x is None:
        return None
    if x.__class__.__module__ == 'torch':
        return x.detach().cpu().numpy()
    if x.__class__.__module__ == 'numpy':
        return x
    return np.array(x)


def load_data(filename):
    with open(filename, 'rb') as fn:
        data_dict = pickle.load(fn)
    num_rows = data_dict['num_rows']
    num_cols = data_dict['num_cols']
    X = data_dict['X']
    Y = data_dict['Y'].squeeze()
    return num_rows, num_cols, X, Y


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
    x, y = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
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

    return grid, y/y.max()


def generate_mixed_data(num_rows, num_cols, num_zs=4, k=4, min_var=.1, max_var=2, algo='sum'):
    x, y = np.meshgrid(np.arange(num_cols), np.arange(num_rows))
    grid = np.vstack([y.flatten(), x.flatten()]).transpose()
    n = num_rows * num_cols
    z_ind = np.random.randint(0, num_zs, n)
    z = np.zeros((n, num_zs))
    z[np.arange(n), z_ind] = 1
    grid = np.concatenate([grid, z], axis=1)
    a, b = grid[:, 0].max(), grid[:, 1].max()
    grid[:, 0] /= a
    grid[:, 1] /= b

    means_x = np.random.uniform(0, num_rows, size=k) / a
    means_y = np.random.uniform(0, num_cols, size=k) / b
    means_z_ind = np.random.randint(0, num_zs, size=k)
    means_z = np.zeros((k, num_zs))
    means_z[np.arange(k), means_z_ind] = 1
    means = np.vstack([means_x, means_y]).transpose()
    means = np.concatenate([means, means_z], axis=1)
    variances = np.random.uniform(min_var, max_var, size=k)

    y = np.zeros(n)
    for i in range(k):
        dist_sq = np.sum(np.square(grid - means[i].reshape(1, -1)), axis=1)
        tmp = np.exp(-dist_sq / variances[i])
        if algo == 'max':
            y = np.maximum(y, tmp)
        elif algo == 'sum':
            y += tmp
    return grid, y/y.max()


def mi_change(x, a, a_bar, gp, x_noise_var=None, a_noise_var=None, a_bar_noise_var=None):
    e1 = conditional_entropy(x, a, gp, x_noise_var, a_noise_var)
    e2 = conditional_entropy(x, a_bar, gp, x_noise_var, a_bar_noise_var)
    info = e1 - e2
    return info


def process_noise_var(dim, noise_var):
    if noise_var is None:
        noise_var_ = 0.0
    elif isinstance(noise_var, int) or isinstance(noise_var, float):
        noise_var_ = noise_var * np.eye(dim)
    elif isinstance(noise_var, list):
        assert len(noise_var) == dim, 'Size mismatch!!'
        noise_var_ = np.diag(noise_var)
    elif isinstance(noise_var, np.ndarray):
        if noise_var.ndim == 1:
            assert len(noise_var) == dim, 'Size mismatch!!'
            noise_var_ = np.diag(noise_var)
        elif noise_var.ndim == 2:
            assert noise_var.shape[0] == noise_var.shape[1] == dim, 'Size mismatch'
            noise_var_ = noise_var
    else:
        raise NotImplementedError
    return noise_var_


def entropy(x, gp, x_noise_var=0):
    x_ = x.reshape(-1, len(x)) if x.ndim == 1 else x

    # NOTE: because of noise term, even if there are repeated entries, the det is not 0
    x_noise_var_ = process_noise_var(x_.shape[0], x_noise_var)
    cov = gp.cov_mat(x_, x_, x_noise_var_)
    return entropy_from_cov(cov)


def entropy_from_cov(cov):
    ent = cov.shape[0] * CONST + .5 * np.linalg.slogdet(cov)[1].item()
    return ent


def conditional_entropy(x, a, gp, x_noise_var, a_noise_var, sigma_aa_inv=None):
    assert a.ndim == 2, 'Matrix A must be 2-dimensional!'
    if a.shape[0] == 0:
        return entropy(x, gp, x_noise_var)

    x_ = x.reshape(-1, a.shape[-1])
    x_noise_var_ = process_noise_var(x_.shape[0], x_noise_var)
    a_noise_var_ = process_noise_var(a.shape[0], a_noise_var)

    if sigma_aa_inv is None:
        sigma_aa_inv = np.linalg.inv(gp.cov_mat(a, a, a_noise_var_))
    sigma_xa = gp.cov_mat(x_, a)
    sigma_xx = gp.cov_mat(x_, x_, x_noise_var_)
    cov = sigma_xx - np.dot(np.dot(sigma_xa, sigma_aa_inv), sigma_xa.T)
    return entropy_from_cov(cov)


def is_valid_cell(cell, grid_shape):
    if 0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]:
        return True
    return False


def vec_to_one_hot_matrix(vec, max_val=None):
    if max_val is None:
        max_val = np.max(vec)
    mat = np.zeros((len(vec), max_val+1))
    mat[np.arange(len(vec)), vec] = 1
    return mat


def zero_mean_unit_variance(data, mean=None, std=None):
    # zero mean unit variance normalization
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    return (data - mean) / std


def draw_plots(num_rows, num_cols, plot1, plot2, plot3, main_title=None,
               title1=None, title2=None, title3=None, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        axt, axp, axv = ax

    title1 = 'Ground truth' if title1 is None else title1
    axt.set_title(title1)
    imt = axt.imshow(plot1.reshape(num_rows, num_cols),
                     cmap='ocean', vmin=plot1.min(), vmax=plot1.max())
    div = make_axes_locatable(axt)
    caxt = div.new_horizontal(size='5%', pad=.05)
    fig.add_axes(caxt)
    fig.colorbar(imt, caxt, orientation='vertical')

    title2 = 'Predicted values' if title2 is None else title2
    axp.set_title(title2)
    imp = axp.imshow(plot2.reshape(num_rows, num_cols),
                     cmap='ocean', vmin=plot1.min(), vmax=plot1.max())
    divm = make_axes_locatable(axp)
    caxp = divm.new_horizontal(size='5%', pad=.05)
    fig.add_axes(caxp)
    fig.colorbar(imp, caxp, orientation='vertical')

    title3 = 'Variance' if title3 is None else title3
    axv.set_title(title3)
    imv = axv.imshow(plot3.reshape(num_rows, num_cols), cmap='hot')
    divv = make_axes_locatable(axv)
    caxv = divv.new_horizontal(size='5%', pad=.05)
    fig.add_axes(caxv)
    fig.colorbar(imv, caxv, orientation='vertical')

    if main_title is not None:
        fig.suptitle(main_title)
    return fig

