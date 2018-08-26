from utils import generate_gaussian_data, zero_mean_unit_variance
from methods import greedy
from models import GpytorchGPR
import numpy as np
import ipdb

# RESULT: scaling up as done in this scipt makes entropy monotonic

num_rows, num_cols = 10, 10
n = num_rows * num_cols
X, Y = generate_gaussian_data(num_rows, num_cols)
X = zero_mean_unit_variance(X)
gp = GpytorchGPR(lr=.01, max_iterations=500)

# train on random 10 points
num_train = 10
perm = np.random.permutation(n)
train_ind = perm[:num_train]
test_ind = perm[num_train:]

x_train = X[train_ind]
y_train = Y[train_ind]
gp.fit(x_train, y_train)

sampled = np.full(n, False)
alpha = 1e-5
cov = gp.cov_mat(X) + alpha * np.eye(n)
num_samples = n

seq1, utilities1 = greedy(X, np.copy(sampled), cov, num_samples)

min_eig = np.linalg.eigvalsh(cov).min()
constant = -.5 * np.log(min_eig)
seq2, utilities2 = greedy(X, np.copy(sampled), cov, num_samples, entropy_constant=constant)
