from models import GPR
from utils import compute_rmse, posterior_distribution
import numpy as np
import ipdb


def baseline(env, args):
	train_x = env.X
	train_y = env.Y
	train_var = np.full(len(env.X), args.sensor_std**2)
	test_x = env.test_X
	test_y = env.test_Y

	# train on the entire training set X
	gp1 = GPR(latent=args.latent, lr=args.lr, max_iterations=args.max_iterations, learn_likelihood_noise=True)
	gp1.fit(train_x, train_y, train_var, disp=False)
	# mu1 = gp1.predict(test_x)
	# rmse1 = compute_rmse(mu1, test_y)
	
	mu, cov = posterior_distribution(gp1, train_x, train_y, test_x, train_var, return_cov=True)
	rmse = compute_rmse(test_y, mu)

	result = {'mean': mu, 'covariance': cov, 'rmse': rmse}
	return result

	# # train on a subset D
	# gp2 = GPR(latent=args.latent, lr=args.lr, max_iterations=args.max_iterations)
	# num_train = int(.75*len(train_x))
	# train_ind = np.random.permutation(len(train_x))[:num_train]
	# gp2.fit(train_x[train_ind], train_y[train_ind], train_var[train_ind], disp=False)
	
	# # condition on D
	# mu2D = gp2.predict(test_x)
	# rmse2D = compute_rmse(mu2D, test_y)

	# mu2Dp = posterior_distribution(gp2, train_x[train_ind], train_y[train_ind], test_x, train_var[train_ind])
	# rmse2Dp = compute_rmse(mu2Dp, test_y)

	# # condition on X
	# gp2.set_train_data(train_x, train_y, train_var)
	# mu2X = gp2.predict(test_x)
	# rmse2X = compute_rmse(mu2X, test_y)

	# mu2Xp = posterior_distribution(gp2, train_x, train_y, test_x, train_var)
	# rmse2Xp = compute_rmse(mu2Xp, test_y)
