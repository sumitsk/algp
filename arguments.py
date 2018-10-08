import os
import argparse
import warnings
import sys


def get_args():
    parser = argparse.ArgumentParser(description='Adaptive Sampling and Informative Planning')

    # gp model 
    parser.add_argument('--lr', default=.1, type=float, help='learning rate of GP model')
    parser.add_argument('--max_iterations', default=200, type=int, help='number of training iterations for GP model')
    parser.add_argument('--data_file', default=None, help='pickle file to load data from')
    parser.add_argument('--phenotype', default='plant_height', help='target feature')
    parser.add_argument('--kernel', default='rbf', help='kernel of GP model {rbf, matern}')
    # parser.add_argument('--n_mixtures', default=4, help='number of spectral mixture components')
    parser.add_argument('--latent', default=None, help='latent function in GP model')
    
    parser.add_argument('--num_sims', default=10, type=int, help='number of simulations')
    parser.add_argument('--num_runs', default=6, type=int, help='number of batches')
    parser.add_argument('--fraction_pretrain', default=.75, type=float, help='fraction of all training data used for learning hyperparameters')
    parser.add_argument('--num_samples_per_batch', default=4, type=int, help='number of static samples collected in each batch')
    parser.add_argument('--slack', default=0, type=int, help='budget = shortest path length + slack')
    parser.add_argument('--num_test', default=40, type=int, help='number of test samples')

    parser.add_argument('--update', action='store_true', help='update gp model')
    parser.add_argument('--update_every', default=1, type=int, help='update gp model every ... batch')
    parser.add_argument('--criterion', default='entropy', help='one from {mutual_information, entropy}')
    parser.add_argument('--mobile_enabled', action='store_true', help='include mobile measurements')
    # parser.add_argument('--mobile_std', default=.5, type=float, help='standard deviation of mobile measurements')
    parser.add_argument('--static_std', default=.1, type=float, help='standard deviation of static measurements')
    
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