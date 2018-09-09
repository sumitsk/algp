import numpy as np
import ipdb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import pickle 
import pandas as pd


CONST = .5*np.log(2*np.pi*np.exp(1))


class Node(object):
    def __init__(self, map_pose, gval, utility, parents_index, path=None):
        self.map_pose = map_pose
        self.gval = gval
        self.utility = utility
        self.parents_index = parents_index[:]
        self.path = np.empty((0, 2)) if path is None else np.copy(path)
        

class BFSNode(object):
    def __init__(self, pose, gval, visited, path=None):
        self.pose = pose
        self.gval = gval
        self.path = [tuple(self.pose)] if path is None else path 
        self.visited = np.copy(visited)


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
    valid = data_dict['valid'].squeeze()
    return num_rows, num_cols, X[valid], Y[valid]


def load_dataframe(filename, target_feature, extra_input_features=[]):
    df = pd.read_pickle(filename)
    num_rows = 15
    num_cols = 37
    X = np.vstack(df[['X']].values.squeeze())
    row_range = X[:, :2]
    gene = X[:, 2:]
    ph_vals = np.hstack([df[[f]].values for f in extra_input_features])
    final_x = np.concatenate([row_range, ph_vals, gene], axis=1)
    y = df[[target_feature]].values.squeeze()
    genotype = df[['category']].values.squeeze()
    return num_rows, num_cols, final_x, y, genotype


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


def mi_change(x, a, a_bar, gp, x_variance=None, a_variance=None, a_bar_variance=None):
    e1 = conditional_entropy(x, a, gp, x_variance, a_variance)
    e2 = conditional_entropy(x, a_bar, gp, x_variance, a_bar_variance)
    info = e1 - e2
    return info


def process_variance(dim, variance):
    if variance is None:
        variance_ = 0.0
    elif isinstance(variance, int) or isinstance(variance, float):
        variance_ = variance * np.eye(dim)
    elif isinstance(variance, list):
        assert len(variance) == dim, 'Size mismatch!!'
        variance_ = np.diag(variance)
    elif isinstance(variance, np.ndarray):
        variance = np.squeeze(variance)
        if variance.ndim == 1:
            assert len(variance) == dim, 'Size mismatch!!'
            variance_ = np.diag(variance)
        elif variance.ndim == 2:
            assert variance.shape[0] == variance.shape[1] == dim, 'Size mismatch'
            variance_ = variance
    else:
        raise NotImplementedError
    return variance_


def entropy(x, gp, x_variance=0):
    x_ = x.reshape(-1, len(x)) if x.ndim == 1 else x

    # NOTE: because of noise term, even if there are repeated entries, the det is not 0
    x_variance_ = process_variance(x_.shape[0], x_variance)
    cov = gp.cov_mat(x_, x_, x_variance_)
    return entropy_from_cov(cov)


def entropy_from_cov(cov, constant=CONST):
    # constant is the first term in entropy calculation
    # H = constant * k + 1/2 * log(det(cov))
    if constant is None:
        constant = CONST
    ent = cov.shape[0] * constant + .5 * np.linalg.slogdet(cov)[1].item()
    return ent


def conditional_entropy(x, a, gp, x_variance, a_variance, sigma_aa_inv=None):
    assert a.ndim == 2, 'Matrix A must be 2-dimensional!'
    if a.shape[0] == 0:
        return entropy(x, gp, x_variance)

    x_ = x.reshape(-1, a.shape[-1])
    x_variance_ = process_variance(x_.shape[0], x_variance)
    a_variance_ = process_variance(a.shape[0], a_variance)

    if sigma_aa_inv is None:
        sigma_aa_inv = np.linalg.inv(gp.cov_mat(a, a, a_variance_))
    sigma_xa = gp.cov_mat(x_, a)
    sigma_xx = gp.cov_mat(x_, x_, x_variance_)
    cov = sigma_xx - np.dot(np.dot(sigma_xa, sigma_aa_inv), sigma_xa.T)
    return entropy_from_cov(cov)


def is_valid_cell(cell, grid_shape):
    # check if cell lies inside the grid or not
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


def normalize(data, col_max=None):
    # divide each column with the corresponding max value
    col_max = data.max(0) if col_max is None else col_max
    return data/col_max


# def draw_plots(num_rows, num_cols, plot1, plot2, plot3, main_title=None,
#                title1=None, title2=None, title3=None, fig=None, ax=None):
#     if fig is None or ax is None:
#         fig, ax = plt.subplots(1, 3, figsize=(12, 4))
#         axt, axp, axv = ax

#     # TODO: use seaborn 
#     title1 = 'Ground truth' if title1 is None else title1
#     axt.set_title(title1)
#     imt = axt.imshow(plot1.reshape(num_rows, num_cols),
#                      cmap='ocean', vmin=plot1.min(), vmax=plot1.max())
#     div = make_axes_locatable(axt)
#     caxt = div.new_horizontal(size='5%', pad=.05)
#     fig.add_axes(caxt)
#     fig.colorbar(imt, caxt, orientation='vertical')

#     title2 = 'Predicted values' if title2 is None else title2
#     axp.set_title(title2)
#     imp = axp.imshow(plot2.reshape(num_rows, num_cols),
#                      cmap='ocean', vmin=plot1.min(), vmax=plot1.max())
#     divm = make_axes_locatable(axp)
#     caxp = divm.new_horizontal(size='5%', pad=.05)
#     fig.add_axes(caxp)
#     fig.colorbar(imp, caxp, orientation='vertical')

#     title3 = 'Variance' if title3 is None else title3
#     axv.set_title(title3)
#     imv = axv.imshow(plot3.reshape(num_rows, num_cols), cmap='hot')
#     divv = make_axes_locatable(axv)
#     caxv = divv.new_horizontal(size='5%', pad=.05)
#     fig.add_axes(caxv)
#     fig.colorbar(imv, caxv, orientation='vertical')

#     if main_title is not None:
#         fig.suptitle(main_title)
#     return fig


def compute_rmse(true, pred):
    # return root mean square error betwee true values and predictions
    return np.linalg.norm(true.squeeze() - pred.squeeze()) / np.sqrt(len(true))


def euclidean_distance(p0, p1):
    # return euclidean distance between p0 and p1
    return ((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)**.5


def manhattan_distance(p0, p1):
    # return manhattan distance between p0 and p1
    return abs(p0[0] - p1[0]) + abs(p0[1] - p1[1])    


def fit_and_eval(gp, train_x, train_y, test_x, test_y, disp=False):
    # fit a gp model and evaluate on the training and testing dataset
    gp.fit(train_x, train_y, disp=disp)
    pred_train = gp.predict(train_x)
    pred_test = gp.predict(test_x)
    train_rmse = compute_rmse(train_y, pred_train)
    test_rmse = compute_rmse(test_y, pred_test)
    return train_rmse, test_rmse


def posterior_distribution(gp, train_x, train_y, test_x, train_var=None, test_var=None, return_var=False, return_cov=False):
    train_y_mean = np.mean(train_y)

    # robust to the behavior of white noise kernel
    cov_aa = gp.cov_mat(x1=train_x, var=train_var, add_likelihood_var=True)
    cov_xx = gp.cov_mat(x1=test_x, var=test_var)
    cov_xa = gp.cov_mat(x1=test_x, x2=train_x)

    mat1 = np.dot(cov_xa, np.linalg.inv(cov_aa))
    mu = np.dot(mat1, (train_y-train_y_mean)) + train_y_mean
    
    if return_var:
        cov = cov_xx - np.dot(mat1, cov_xa.T)
        return mu, np.diag(cov)
         
    if return_cov:
        cov = cov_xx - np.dot(mat1, cov_xa.T)
        return mu, cov
    return mu 


def valid_neighbors(pose, grid_shape, dxdy=None):
    if dxdy is None:
        dxdy=[(0,1), (0,-1), (1,0), (-1,0)]
    nodes = []
    for dx,dy in dxdy:
        new_node = (pose[0] + dx, pose[1] + dy) 
        if is_valid_cell(new_node, grid_shape):
            nodes.append(new_node)
    return nodes


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


def get_monotonic_entropy_constant(cov_matrix):
    eig_vals = np.linalg.eigvalsh(cov_matrix)
    min_eig = min(eig_vals)
    entropy_constant = -.5 * np.log(min_eig)
    return entropy_constant


def draw_path(ax, path, head_width=None, head_length=None, linewidth=None, delta=None, color=None):
    head_width = .05 if head_width is None else head_width
    head_length = .1 if head_length is None else head_length
    linewidth = 2 if linewidth is None else linewidth
    delta = head_length*2 if delta is None else delta
    arrow_color = 'red' if color is None else color

    for i in range(len(path)-1):
        source = path[i]
        sink = path[i+1]
        dxdy = (sink[0]-source[0], sink[1]-source[1])
        dx = dxdy[0]
        dy = dxdy[1]
        if dx == 0:
            sign = dy//abs(dy)
            dy = sign*(abs(dy)-delta)
        else:
            sign = dx//abs(dx)
            dx = sign*(abs(dx)-delta)
        
        ax.arrow(source[0], source[1], dx, dy,
                 head_width=head_width, head_length=head_length,
                 linewidth=linewidth, color=arrow_color, alpha=1)