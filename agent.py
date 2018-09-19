import ipdb
import numpy as np
import torch 

from utils import entropy_from_cov, compute_rmse, posterior_distribution, get_monotonic_entropy_constant, find_shortest_path, find_equi_sample_path
from graph_utils import get_heading
from models import GPR
import time


class Agent(object):
    def __init__(self, env, args, parent_agent=None, learn_likelihood_noise=True, mobile_std=None, static_std=None):
        super()
        self.env = env
        self.learn_likelihood_noise = learn_likelihood_noise
        self._init_model(args)

        self.mobile_std = args.mobile_std if mobile_std is None else mobile_std
        self.static_std = args.static_std if static_std is None else static_std
        self.num_samples_per_batch = args.num_samples_per_batch
        self.update_every = args.update_every
        
        self.reset()
        if parent_agent is None:
            num_pretrain = int(args.fraction_pretrain * self.env.num_samples)
            # num_pretrain = args.num_pretrain
            self._pre_train(num_samples=num_pretrain)
        else:
            self.load_model(parent_agent)

    def _init_model(self, args):
        kernel_params = self._get_kernel_params(args)
        self.gp = GPR(latent=args.latent, lr=args.lr, max_iterations=args.max_iterations, kernel_params=kernel_params,
                      learn_likelihood_noise=self.learn_likelihood_noise)
        
    def _get_kernel_params(self, args):
        kernel_params = {'type': args.kernel}
        if args.kernel == 'spectral_mixture':
            kernel_params['n_mixtures'] = args.n_mixtures
        return kernel_params

    def load_model(self, parent_agent):
        self.gp.reset(parent_agent.gp.train_x, parent_agent.gp.train_y, parent_agent.gp.train_var)
        self.gp.model.load_state_dict(parent_agent.gp.model.state_dict())
        
    def save_model(self, filename):
        state = {'state_dict': self.gp.model.state_dict()}
        torch.save(state, filename)

    def reset(self):
        self.static_data = [[] for _ in range(self.env.num_samples)]
        self.mobile_data = [[] for _ in range(self.env.num_samples)]
        self.pose = (0, 0)
        self.heading = (1, 0)
        self.path = np.copy(self.pose).reshape(-1, 2)
        self.static_locations = np.empty((0, 2))
        
    def _pre_train(self, num_samples):
        print('====================================================')
        print('--- Pretraining ---')
        ind = np.random.permutation(self.env.num_samples)[:num_samples]
        self._add_samples(ind, source='static')
        self.update_model()
        # if don't want to remember/condition on pre_train data points    
        # self.reset()    
        
    def _add_samples(self, indices, source='static'):
        std = self.static_std if source == 'static' else self.mobile_std
        y = self.env.collect_samples(indices, std)
        for i in range(len(indices)):
            idx = indices[i]
            if source == 'static':
                self.static_data[idx].append(y[i])
            else:
                self.mobile_data[idx].append(y[i])

    def update_model(self):
        indices, y, var = self.get_sampled_dataset()        
        x = self.env.X[indices]
        self.gp.fit(x, y, var)
        
    def get_sampled_dataset(self):
        all_y = []
        all_var = []
        indices = []
        for i in range(self.env.num_samples):
            if len(self.mobile_data[i])>0 and len(self.static_data[i])>0:
                yc = np.mean(self.mobile_data[i])
                ys = np.mean(self.static_data[i])
                yeq = (self.mobile_std**2 * ys + self.static_std**2 * yc) / (self.mobile_std**2 + self.static_std**2)
                var = 1 / (1/(self.static_std**2) + 1/(self.mobile_std**2))

            elif len(self.static_data[i])>0:
                yeq = np.mean(self.static_data[i])
                var = self.static_std**2

            elif len(self.mobile_data[i])>0:
                yeq = np.mean(self.mobile_data[i])
                var = self.mobile_std**2

            else:
                continue
            all_y.append(yeq)
            all_var.append(var)
            indices.append(i)

        return indices, np.array(all_y), np.array(all_var)

    def _post_update(self):
        # add small value to diagonal (for example constant likelihood noise)
        self.cov_matrix = self.gp.cov_mat(x1=self.env.X, add_likelihood_var=True)
        if self.criterion == 'monotonic_entropy':
            self.entropy_constant = get_monotonic_entropy_constant(self.cov_matrix)
        else:
            self.entropy_constant = None

    def _setup_ipp(self, criterion, update):
        self.criterion = criterion
        if not update:
            self.reset()
        self._post_update()

    def run_ipp(self, render=False, num_runs=10, criterion='entropy', mobile_enabled=False, update=False, slack=0, strategy='max_ent'):
        # this function selects k points greedily and then finds the path that maximises the total information gain 
        assert strategy in ['max_ent', 'shortest', 'equi_sample'], 'Unknown strategy!!'
        assert criterion in ['entropy', 'mutual_information'], 'Unknown criterion!!'
        self._setup_ipp(criterion, update)

        least_cost_paths = 0
        test_rmse = []

        for i in range(num_runs):
            print('\n==================================================================================================')
            print('Run {}/{}'.format(i+1, num_runs))
            run_start = time.time()
            
            # greedily select static samples
            new_gp_indices = self.greedy(self.num_samples_per_batch)  
            waypoints = [tuple(self.env.gp_index_to_map_pose(x)) for x in new_gp_indices]
            next_static_locations = np.stack(waypoints)
            self.static_locations = np.concatenate([self.static_locations, next_static_locations]).astype(int)
            
            # Gather data along path only when mobile is enabled
            if mobile_enabled:          
                print('------ Finding valid paths ---------')
                print('Pose:',self.pose, 'Heading:', self.heading, 'Waypoints:', waypoints)
                start = time.time()

                least_cost_ub = self.env.get_heuristic_cost(self.pose, self.heading, waypoints)
                print('Least cost upper bound:',least_cost_ub)
                
                paths_checkpoints, paths_indices, paths_cost = self.env.get_shortest_path_waypoints(self.pose, self.heading, waypoints, least_cost_ub, slack)
                end = time.time()
                print('Number of feasible paths: ', len(paths_indices))
                print('Time consumed {:.4f}'.format(end - start))

                print('\n------ Finding best path ----------')
                start = time.time()
                if strategy == 'shortest':
                    best_idx = find_shortest_path(paths_cost)
                else:
                    best_idx = self.best_path(paths_indices, new_gp_indices)
                    if strategy == 'equi_sample':
                        best_idx = find_equi_sample_path(paths_indices, best_idx)

                next_path_indices = paths_indices[best_idx]
                next_path = np.stack(self.env.get_path_from_checkpoints(paths_checkpoints[best_idx]))
                end = time.time()
                least_cost = min(paths_cost)
                print('Least cost: {} Best path cost: {}'.format(least_cost, paths_cost[best_idx]))
                print('Time consumed {:.4f}'.format(end - start))
                
                if least_cost == paths_cost[best_idx]:
                    least_cost_paths += 1

                if render:
                    self.env.render(paths_checkpoints[best_idx], self.path, next_static_locations, self.static_locations)
                
                # update agent statistics
                self.path = np.concatenate([self.path, next_path[1:]], axis=0).astype(int)
                self.pose = tuple(self.path[-1])
                self.heading = get_heading(self.path[-1], self.path[-2])
        
                # add samples
                self._add_samples(next_path_indices, source='mobile')
            
            self._add_samples(new_gp_indices, source='static')
            
            # update hyperparameters of GP model
            if update and (i+1) % self.update_every == 0:
                print('\n---------- Updating model --------------')
                start = time.time()
                self.update_model()
                self._post_update()
                end = time.time()
                print('Time consumed {:.4f}'.format(end - start))

            print('\n-------- Prediction -------------- ')
            start = time.time()
            pred, cov = self.predict_test()
            var = np.diag(cov)
            rmse = compute_rmse(self.env.test_Y, pred)
            print('RMSE: {:.4f}'.format(rmse))
            print('Predictive Variance Max: {:.3f} Min: {:.3f} Mean: {:.3f}'.format(var.max(), var.min(), var.mean()))
            end = time.time()
            print('Time consumed {:.4f}'.format(end - start))

            # log predictions
            # res = {'mean': pred, 'covariance': cov, 'rmse': rmse}
            test_rmse.append(rmse)
            
            run_end = time.time()
            print('\nTotal Time consumed in run {}: {:.4f}'.format(i+1, run_end - run_start))

        print('==========================================================')
        print('--- Final statistics --- ')
        print('RMSE: {:.4f}'.format(rmse))
        print('Predictive Variance Max: {:.3f} Min: {:.3f} Mean: {:.3f}'.format(var.max(), var.min(), var.mean()))

        # results = {'least_cost_paths': least_cost_paths, 'rmse': test_rmse, 'mobile_samples_count': mobile_samples_count}
        results = {'mean': pred, 'rmse':test_rmse}
        return results

    # def predict_train(self, test_ind=None):
    #     test_ind = np.arange(self.env.num_samples) if test_ind is None else test_ind
    #     train_ind, train_y, train_var = self.get_sampled_dataset()

    #     cov_aa = self.cov_matrix[train_ind].T[train_ind].T + np.diag(train_var)
    #     cov_xx = self.cov_matrix[test_ind].T[test_ind].T
    #     cov_xa = self.cov_matrix[test_ind].T[train_ind].T

    #     mat1 = np.dot(cov_xa, np.linalg.inv(cov_aa))
    #     mu = np.dot(mat1, train_y.reshape(-1,1))
    #     cov = cov_xx - np.dot(mat1, cov_xa.T)
    #     return mu, np.diag(cov)

    def predict_test(self):
        train_ind, train_y, train_var = self.get_sampled_dataset()
        train_x = self.env.X[train_ind]
        mu, cov = posterior_distribution(self.gp, train_x, train_y, self.env.test_X, train_var, return_cov=True)
        return mu, cov

    def greedy(self, num_samples):
        # select most informative samples in a greedy manner
        n = self.env.num_samples
        mobile_sampled = np.array([False if len(x)==0 else True for x in self.mobile_data])
        mobile_var = np.full(n, np.inf)
        mobile_var[mobile_sampled] = self.mobile_std**2
        
        static_sampled = np.array([False if len(x)==0 else True for x in self.static_data])
        static_var = np.full(n, np.inf)
        static_var[static_sampled] = self.static_std**2
        
        sampled = static_sampled | mobile_sampled
        var = 1.0 / (1.0/static_var[sampled] + 1.0/mobile_var[sampled])
        cov_v = self.cov_matrix[sampled].T[sampled].T + np.diag(var)
        ent_v = entropy_from_cov(cov_v, self.entropy_constant)

        cumm_utilities = []
        new_samples = []
        for _ in range(num_samples):
            utilities = np.full(n, -np.inf)
            cond = ent_v + sum(cumm_utilities)

            for i in range(n):
                if static_sampled[i]:
                    continue

                # modify sampled (temporarily)
                static_sampled[i] = True
                static_var[i] = self.static_std**2
                sampled = static_sampled | mobile_sampled
                var = 1.0 / (1.0/static_var[sampled] + 1.0/mobile_var[sampled])
        
                # a - set of all sampled locations 
                cov_a = self.cov_matrix[sampled].T[sampled].T + np.diag(var)
                ent_a = entropy_from_cov(cov_a, self.entropy_constant)
                if self.criterion == 'mutual_information':
                    cov_abar = self.cov_matrix[~sampled].T[~sampled].T 
                    ent_abar = entropy_from_cov(cov_abar)
                    
                    precision = 1.0/static_var + 1.0/mobile_var
                    precision[precision==0] = np.inf
                    var = 1.0 / precision
                    cov_all = self.cov_matrix + np.diag(var)
                    ent_all = entropy_from_cov(cov_all)
                    ut = ent_a + ent_abar - ent_all
                else:
                    ut = ent_a - cond

                utilities[i] = ut

                # reset sampled
                static_sampled[i] = False
                static_var[i] = np.inf

            best_sample = np.argmax(utilities)
            cumm_utilities.append(utilities[best_sample])
            new_samples.append(best_sample)
            # update sampled
            static_sampled[best_sample] = True
            static_var[best_sample] = self.static_std**2
 
        return new_samples

    def best_path(self, paths_mobile_indices, static_indices):
        # paths_indices contains mobile sensing indices on the path
        # static_indices is the set of static sensing indices 

        if len(paths_mobile_indices) == 1:
            return 0

        n = self.env.num_samples
        org_mobile_sampled = np.array([False if len(x)==0 else True for x in self.mobile_data])
        
        static_sampled = np.array([False if len(x)==0 else True for x in self.static_data])
        static_sampled[static_indices] = True
        static_var = np.full(n, np.inf)
        static_var[static_sampled] = self.static_std**2
        
        all_ut = []
        for i in range(len(paths_mobile_indices)):
            mobile_sampled = np.copy(org_mobile_sampled)
            mobile_indices = paths_mobile_indices[i]
            mobile_sampled[mobile_indices] = True
            
            mobile_var = np.full(n, np.inf)
            mobile_var[mobile_sampled] = self.mobile_std**2
        
            sampled = static_sampled | mobile_sampled
            var = 1.0 / (1.0/static_var[sampled] + 1.0/mobile_var[sampled])
        
            # a - set of all sampled locations 
            cov_a = self.cov_matrix[sampled].T[sampled].T + np.diag(var)
            ent_a = entropy_from_cov(cov_a, self.entropy_constant)
            if self.criterion == 'mutual_information':
                cov_abar = self.cov_matrix[~sampled].T[~sampled].T 
                ent_abar = entropy_from_cov(cov_abar)
                
                precision = 1.0/static_var + 1.0/mobile_var
                precision[precision==0] = np.inf
                var = 1.0 / precision
                cov_all = self.cov_matrix + np.diag(var)
                ent_all = entropy_from_cov(cov_all)
                ut = ent_a + ent_abar - ent_all
            else:
                ut = ent_a
            all_ut.append(ut)

        idx = np.argmax(all_ut)
        return idx

        