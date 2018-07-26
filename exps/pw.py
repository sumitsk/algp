import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ipdb

from env import FieldEnv
from models import GpytorchGPR, SklearnGPR
from utils import draw_plots, zero_mean_unit_variance

def draw_3_plots(num_rows, num_cols, true_y, y1, y2,
               title=None, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        axt, axp, axv = ax

    axt.set_title('Ground truth')
    imt = axt.imshow(true_y.reshape(num_rows, num_cols),
                     cmap='ocean', vmin=true_y.min(), vmax=true_y.max())
    div = make_axes_locatable(axt)
    caxt = div.new_horizontal(size='5%', pad=.05)
    fig.add_axes(caxt)
    fig.colorbar(imt, caxt, orientation='vertical')

    axp.set_title('Predicted values from simple GP')
    imp = axp.imshow(y1.reshape(num_rows, num_cols),
                     cmap='ocean', vmin=true_y.min(), vmax=true_y.max())
    divm = make_axes_locatable(axp)
    caxp = divm.new_horizontal(size='5%', pad=.05)
    fig.add_axes(caxp)
    fig.colorbar(imp, caxp, orientation='vertical')

    axv.set_title('Predicted values from GP with gene embedding')
    imv = axv.imshow(y2.reshape(num_rows, num_cols),
    				 cmap='ocean', vmin=true_y.min(), vmax=true_y.max())
    divv = make_axes_locatable(axv)
    caxv = divv.new_horizontal(size='5%', pad=.05)
    fig.add_axes(caxv)
    fig.colorbar(imv, caxv, orientation='vertical')

    if title is not None:
        fig.suptitle(title)
    return fig


def transform_y(y, inverse=False):
    # return y
    if not inverse:
        return np.log(y + 1)
    return np.exp(y) - 1


def fit_and_eval(gp, x, y, x_train, y_train, **kwargs):
    var = np.full(len(y_train), .05)
    # var = None
    gp.fit(x_train, y_train, var, **kwargs)
    mu, std = gp.predict(x, return_std=True)
    rmse = compute_rmse(y, mu)
    return mu, std, rmse

def evaluate(gp, x, y):
	mu = gp.predict(x)
	rmse = compute_rmse(y, mu)
	return rmse

def compute_rmse(y, mu):
	return np.linalg.norm(y - mu) / np.sqrt(len(y))

# file with field data
feature = 'plant_width_mean'
# feature = 'plant_height_mean(cm)'
# feature = 'height_aerial(cm)'
# feature = 'plant_count_mean'
data_file = 'data/' + feature + '_dataset.pkl'
env = FieldEnv(data_file=data_file)

num_rows, num_cols = env.shape
n = env.num_samples
num_train = int(.6*n)
train_indices = np.random.randint(0, n, num_train)
test_indices = np.array(list(set(range(n)) - set(train_indices)))

noise_std = .05
noise_var = np.full(num_train, noise_std**2)
x = env.X
y = env.Y

# normalize dataset
# x[:, :2] = zero_mean_unit_variance(x[:, :2])
# x[: ,0] *= 2
x[:, :2] /= x[:, :2].max(axis=0)

x_train = x[train_indices, :]
y_train = y[train_indices]
x_test = x[test_indices, :]
y_test = y[test_indices]

# y_train = env.collect_samples(train_indices, noise_std)
# y_train = transform_y(y_train)

# gp1 = SklearnGPR()
# mu1, std1, rmse1 = fit_and_eval(gp1, x, y, x_train, y_train)
# test_rmse1 = evaluate(gp1, x_test, y_test)

lr = .1
gp2 = GpytorchGPR(latent=None, lr=lr, kernel='matern')
kwargs2 = {}
mu2, std2, rmse2 = fit_and_eval(gp2, x, y, x_train, y_train, **kwargs2)
test_rmse2 = evaluate(gp2, x_test, y_test)

# gp3 = GpytorchGPR(latent='linear', lr=lr)
# kwargs3 = {'input_dim': 6, 'embed_dim': 3}
# mu3, std3, rmse3 = fit_and_eval(gp3, x, y, x_train, y_train, **kwargs3)
#
# gp4 = GpytorchGPR(latent='non_linear', lr=lr)
# kwargs4 = {'input_dim': 6, 'f1_dim': 3, 'f2_dim': 3}
# mu4, std4, rmse4 = fit_and_eval(gp4, x, y, x_train, y_train, **kwargs4)

gp_field = GpytorchGPR(latent='field', lr=lr)
kwargs_field = {'rr_dim': 2, 'v_dim': x.shape[-1]-2}
mu_field, std_field, rmse_field = fit_and_eval(gp_field, x, y, x_train, y_train, **kwargs_field)
test_rmse_field = evaluate(gp_field, x_test, y_test)

# print('GP_sklearn RMSE', test_rmse1)
print('GP_simple RMSE', test_rmse2)
# print('GP_linear RMSE', rmse3)
# print('GP_non_linear RMSE', rmse4)
print('GP_field', test_rmse_field)

# fig1 = draw_plots(num_rows, num_cols, y, mu1, std1**2, 'GP_sklearn')
fig2 = draw_plots(num_rows, num_cols, y, mu2, std2**2, 'GP_simple')
# fig3 = draw_plots(num_rows, num_cols, y, mu3, std3**2, 'GP_linear')
# fig4 = draw_plots(num_rows, num_cols, y, mu4, std4**2, 'GP_non_linear')
# fig5 = draw_plots(num_rows, num_cols, y, mu_field, std_field**2, 'GP_field')

# fig = draw_3_plots(num_rows, num_cols, y, mu2, mu_field, feature)
plt.show()


