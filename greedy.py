from utils import generate_gaussian_data, zero_mean_unit_variance
from models import GpytorchGPR
import numpy as np
import ipdb

CONST = .5*np.log(2*np.pi*np.exp(1))


def distance(p1, p2):
	return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])


def greedy(X, sampled, K, locs, pose, r=np.inf):
	K_inv = np.linalg.inv(K[sampled].T[sampled].T)

	n = X.shape[0]
	best_ind = 0
	max_var = 0
	for i in range(n):
		if not sampled[i] and distance(pose, locs[i]) <= r:
			k = K[sampled, i]
			var = 1.0 - np.dot(k.T, np.dot(K_inv, k))
			if var > max_var:
				best_ind = i
				max_var = var
	# NOTE: validate the monotonicity assumption of entropy
	# after a while, entropy change becomes negative 
	max_entropy = CONST + .5 * np.log(max_var)
	return best_ind, max_entropy


def load_data():
	X, Y = generate_gaussian_data(num_rows=40, num_cols=40, k=20, algo='max')
	locs = X[:, :2]
	return X, Y, locs

# experiments TODO:
# 1. radius constrained plot of change in entropy
# 2. min eigenvalue plot of covariance matrix 

if __name__ == '__main__':
	# load data
	X, Y, locs = load_data()
	X = zero_mean_unit_variance(X)

	# initialise model
	gp = GpytorchGPR(lr=.1)

	# pretrain model (pilot survey)
	n = X.shape[0]
	sampled = np.full(n, False)
	ntrain = 1
	train_ind = np.random.randint(low=0, high=n, size=(ntrain))
	# sampled[train_ind] = True
	xtrain = X[train_ind, :]
	ytrain = Y[train_ind]
	gp.fit(xtrain, ytrain, var=None)
	K = gp.cov_mat(X, X)

	pose = [0, 0]
	num_samples = 30
	uc_sampled = np.copy(sampled)
	# unconstrained BO
	for i in range(num_samples):
		next_ind, utility_change = greedy(X, uc_sampled, K, locs, pose)
		uc_sampled[next_ind] = True
		pose = locs[next_ind]
		print(pose, utility_change)

	# constrained BO
	radius = 30
	c_sampled = np.copy(sampled)
	print('\n')
	for i in range(num_samples):
		next_ind, utility_change = greedy(X, c_sampled, K, locs, pose, r=radius)
		c_sampled[next_ind] = True
		pose = locs[next_ind]
		print(pose, utility_change)