import os
import argparse
import warnings
import sys


def get_args():
    parser = argparse.ArgumentParser(description='Adaptive Sampling and Informative Planning')

    # gp model 
    parser.add_argument('--lr', default=.01, type=float, help='learning rate of gpytorch model')
    parser.add_argument('--max_iterations', default=500, type=int, help='number of training iterations for GP model')
    parser.add_argument('--data_file', default=None, help='pickle file to load data from')
    parser.add_argument('--phenotype', default='plant_height', help='target feature')
    parser.add_argument('--kernel', default='rbf', help='kernel of GP model {rbf, matern, spectral_mixture}')
    # parser.add_argument('--n_mixtures', default=4, help='number of spectral mixture components')
    parser.add_argument('--latent', default=None, help='latent function in gpytorch model')
    
    parser.add_argument('--num_runs', default=10, type=int, help='number of batches')
    parser.add_argument('--fraction_pretrain', default=.75, type=float, help='number of samples in pilot survey for model initialization')
    parser.add_argument('--num_samples_per_batch', default=5, type=int, help='number of samples collected by sensor in each batch')
    parser.add_argument('--update_every', default=1, type=int, help='update gp model every ... batch')
    parser.add_argument('--budget_factor', default=1, type=float, help='budget = budget_factor * shortest path length')
    parser.add_argument('--num_test', default=60, type=int, help='number of test samples')

    parser.add_argument('--adaptive', action='store_true', help='update gp model')
    parser.add_argument('--criterion', default='monotonic_entropy', help='one from {mutual_information, entropy, monotonic_entropy}')
    parser.add_argument('--camera_enabled', action='store_true', help='include camera measurements')
    parser.add_argument('--camera_std', default=2.00, type=float, help='standard deviation of camera measurements')
    parser.add_argument('--sensor_std', default=1.00, type=float, help='standard deviation of sensor measurements')
    
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--id', default=1, type=int, help='unique id of every instance')
    parser.add_argument('--save_dir', default='results', help='save directory')
    parser.add_argument('--eval_only', action='store_true', help='will not save anything in this setting')

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, str(args.id))
    if not args.eval_only:
        if os.path.exists(args.save_dir):
            warnings.warn('SAVE DIRECTORY ALREADY EXISTS!')
            ch = input('Press c to continue and s to stop: ')
            if ch == 's':
                sys.exit(0)
            elif ch == 'c':
                os.rename(args.save_dir, args.save_dir+'_old')
            elif ch != 'c':
                raise NotImplementedError 

        os.makedirs(args.save_dir)               
    return args