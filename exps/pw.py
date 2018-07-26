from env import FieldEnv
from models import GpytorchGPR, SklearnGPR
import numpy as np 
import ipdb
from utils import draw_plots, zero_mean_unit_variance
import matplotlib.pyplot as plt


def transform_y(y, inverse=False):
    # return y
    if not inverse:
        return np.log(y + 1)
    return np.exp(y) - 1


def fit_and_eval(gp, x, y, x_train, y_train, **kwargs):
    gp.fit(x_train, y_train, **kwargs)
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
# data_file = 'data/plant_width_mean_dataset.pkl'
# data_file = 'data/plant_height_mean(cm)_dataset.pkl'
# data_file = 'data/height_aerial(cm)_dataset.pkl'
data_file = 'data/plant_count_mean_dataset.pkl'
env = FieldEnv(data_file=data_file)

num_rows, num_cols = env.shape
n = env.num_samples
num_train = int(.7*n)
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

gp1 = SklearnGPR()
mu1, std1, rmse1 = fit_and_eval(gp1, x, y, x_train, y_train)
test_rmse1 = evaluate(gp1, x_test, y_test)

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

print('GP_sklearn RMSE', test_rmse1)
print('GP_simple RMSE', test_rmse2)
# print('GP_linear RMSE', rmse3)
# print('GP_non_linear RMSE', rmse4)
print('GP_field', test_rmse_field)

fig1 = draw_plots(num_rows, num_cols, y, mu1, std1**2, 'GP_sklearn')
fig2 = draw_plots(num_rows, num_cols, y, mu2, std2**2, 'GP_simple')
# fig3 = draw_plots(num_rows, num_cols, y, mu3, std3**2, 'GP_linear')
# fig4 = draw_plots(num_rows, num_cols, y, mu4, std4**2, 'GP_non_linear')
fig5 = draw_plots(num_rows, num_cols, y, mu_field, std_field**2, 'GP_field')
plt.show()

