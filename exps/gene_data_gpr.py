import pickle
from utils import zero_mean_unit_variance, fit_and_eval, normalize
from models import GpytorchGPR
import numpy as np
import ipdb
import matplotlib.pyplot as plt
import seaborn as sns


filename = 'data/gene_data/plant_width_mean_dataset.pkl'
with open(filename, 'rb') as fn:
	dct = pickle.load(fn)

valid = dct['valid']
x = dct['X'][valid]
y = dct['Y'][valid]

x[:, :2] = zero_mean_unit_variance(x[:, :2])
y = normalize(y)

# split into training and test set
n = len(x)
num_train = int(.7*n)
perm = np.random.permutation(n)
train_ind = perm[:num_train]
test_ind = perm[num_train:]

x_train = x[train_ind]
y_train = y[train_ind]
x_test = x[test_ind]
y_test = y[test_ind]

latent = 'non_linear'
latent_params = {'input_dim': x.shape[-1], 'embed_dim': 4}
gp_latent = GpytorchGPR(lr=.05, max_iterations=500)
rmse_train_latent, rmse_test_latent = fit_and_eval(gp_latent, x_train, y_train, x_test, y_test, disp=False)

gp = GpytorchGPR(lr=.05, max_iterations=500)
rmse_train, rmse_test = fit_and_eval(gp, x_train[:,:2], y_train, x_test[:,:2], y_test, disp=False)

gp_female = GpytorchGPR(lr=.05, max_iterations=500)
cols = [0,1,6,7,8,9]
rmse_train_female, rmse_test_female = fit_and_eval(gp_female, x_train[:,cols], y_train, x_test[:, cols], y_test, disp=False)

print('RR:                 ', rmse_train, rmse_test)
print('RR + male + female: ', rmse_train_latent, rmse_test_latent)
print('RR + female:        ', rmse_train_female, rmse_test_female)
ipdb.set_trace()
