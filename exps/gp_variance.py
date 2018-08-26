import math
import numpy as np
import matplotlib.pyplot as plt

from utils import generate_gaussian_data, draw_plots, zero_mean_unit_variance
from models import SklearnGPR, GpytorchGPR
import ipdb

# training data
num_rows = 20
num_cols = 20
n = num_rows * num_cols
x, y = generate_gaussian_data(num_rows, num_cols)
x = zero_mean_unit_variance(x)

# num_train = int(.0/6*n)
num_train = 10
train_indices = np.random.randint(0, n, num_train)
x_train = x[train_indices, :] 
y_train = y[train_indices] + np.random.normal(0, .2, num_train)

# RESULT: 
# 1. sklearn GPR fails poorly in this setting for small as well as large dataset
# from previous exps, it seems that it is highly sensitive to length scale and chagnes abruptly
# 2. Gpytorch GPR is able to perform better than sklearn one in both the cases
# with zero mean, training is stable but number of iterations should be set accordingly

def fit_and_eval(gp, x, y, x_train, y_train, **kwargs):
	gp.fit(x_train, y_train, **kwargs)
	mu, std = gp.predict(x, return_std=True)
	rmse = np.linalg.norm(mu-y)/np.sqrt(len(y))
	return mu, std, rmse

# sklearn GP
gp1 = SklearnGPR()
mu1, std1, rmse1 = fit_and_eval(gp1, x, y, x_train, y_train)

# gpytorch GP
gp2 = GpytorchGPR(latent=None, lr=.1)
kwargs1 = {}
mu2, std2, rmse2 = fit_and_eval(gp2, x, y, x_train, y_train, **kwargs1)

# gp3 = GpytorchGPR(latent='linear', lr=.001)
# kwargs2 = {'input_dim': 2, 'embed_dim': 2}
# mu3, std3, rmse3 = fit_and_eval(gp3, x, y, x_train, y_train, **kwargs2)

# gp4 = GpytorchGPR(latent='non_linear', lr=.001)
# kwargs3 = {'input_dim': 2}
# mu4, std4, rmse4 = fit_and_eval(gp4, x, y, x_train, y_train, **kwargs3)

# print('GP_simple RMSE', rmse2)
# print('GP_linear RMSE', rmse3)
# print('GP_non_linear RMSE', rmse4)

fig1 = draw_plots(num_rows, num_cols, y, mu1, std1**2, 'GP_sklearn')
fig2 = draw_plots(num_rows, num_cols, y, mu2, std2**2, 'GP_simple')
# fig3 = draw_plots(num_rows, num_cols, y, mu3, std3**2, 'GP_linear')
# fig4 = draw_plots(num_rows, num_cols, y, mu4, std4**2, 'GP_non_linear')
plt.show()

ipdb.set_trace()