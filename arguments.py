import os
import argparse
import math

def get_args():
	parser = argparse.ArgumentParser(description='Adaptive Sampling and Informative Planning')

	parser.add_argument('--data_file', default=None, help='pickle file to load data from')
	parser.add_argument('--num_runs', default=20, help='maximum number of path segments')
	parser.add_argument('--norm_factor', default=1, help='divide all observations by this factor! (normalized observations)')

	# gp model 
	parser.add_argument('--model_type', default='gpytorch_GP', help='one from {gpytorch_GP, sklearn_GP}')
	parser.add_argument('--lr', default=.01, help='learning rate of gpytorch model')
	parser.add_argument('--max_iterations', default=200, help='number of training iterations')
	parser.add_argument('--latent', default=None, help='latent function in gpytorch model')
	parser.add_argument('--kernel', default='rbf', help='kernel of GP model {rbf, matern}')
	
	# 
	parser.add_argument('--utility', default='mutual_information', help='one from {mutual_information, entropy}')
	parser.add_argument('--strategy', default='informative', help='only supports informative for now!!')
	parser.add_argument('--precision_method', default='max', help='one from {max, sum}')
	parser.add_argument('--camera_noise', default=1.0, help='variance of camera measurements')
	parser.add_argument('--sensor_noise', default=.05, help='variance of sensor measurements')
	parser.add_argument('--num_pretrain_samples', default=10, help='number of samples in pilot survey for model initialization')

	parser.add_argument('--search_radius', default=10, help='radius of neighborhood')
	parser.add_argument('--mi_radius', default=math.inf, help='radius of ')

	parser.add_argument('--render', action='store_true')
	parser.add_argument('--update_every', default=1, help='update gp model every ... paths planned')

	args = parser.parse_args()
	# TODO: add save directory, to store all results and args
	return args