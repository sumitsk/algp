import ipdb
import numpy as np
import torch 

from utils import entropy_from_cov, compute_rmse, posterior_distribution, get_monotonic_entropy_constant
from graph_utils import get_heading
from models import GPR
import time


class Agent(object):
    def __init__(self, env, args, parent_agent=None, learn_likelihood_noise=False):
        super()
        self.env = env
        self.learn_likelihood_noise = learn_likelihood_noise
        self._init_model(args)

        self.camera_std = args.camera_std
        self.sensor_std = args.sensor_std
        self.num_samples_per_batch = args.num_samples_per_batch
        self.beta = args.budget_factor
        self.update_every = args.update_every
        
        self.reset()
        if parent_agent is None:
            num_pretrain = int(args.fraction_pretrain * self.env.num_samples)
            self._pre_train(num_samples=num_pretrain)
        else:
            self.load_model(parent_agent)

    def _init_model(self, args):
        kernel_params = self._get_kernel_params(args)
        self.gp = GPR(latent=args.latent, lr=args.lr, max_iterations=args.max_iterations, kernel_params=kernel_params,
                      learn_likelihood_noise=self.learn_likelihood_noise)
        
    def load_model(self, parent_agent):
        self.gp.reset(parent_agent.gp.train_x, parent_agent.gp.train_y, parent_agent.gp.train_var)
        self.gp.model.load_state_dict(parent_agent.gp.model.state_dict())
        
    def save_model(self, filename):
        state = {'state_dict': self.gp.model.state_dict()}
        torch.save(state, filename)

    def _get_kernel_params(self, args):
        kernel_params = {'type': args.kernel}
        if args.kernel == 'spectral_mixture':
            kernel_params['n_mixtures'] = args.n_mixtures
        return kernel_params

    def reset(self):
        self.sensor_data = [[] for _ in range(self.env.num_samples)]
        self.camera_data = [[] for _ in range(self.env.num_samples)]
        self.pose = (0, 0)
        self.heading = (1, 0)
        self.path = np.copy(self.pose).reshape(-1, 2)
        self.sensor_locations = np.empty((0, 2))
        
    def _pre_train(self, num_samples):
        print('====================================================')
        print('--- Pretraining ---')
        ind = np.random.permutation(self.env.num_samples)[:num_samples]
        self._add_samples(ind, source='sensor')
        self.update_model()
        # if don't want to remember/condition on pre_train data points    
        self.reset()    
        
    def _add_samples(self, indices, source='sensor'):
        std = self.sensor_std if source == 'sensor' else self.camera_std
        y = self.env.collect_samples(indices, std)
        for i in range(len(indices)):
            idx = indices[i]
            if source == 'sensor':
                self.sensor_data[idx].append(y[i])
            else:
                self.camera_data[idx].append(y[i])

    def update_model(self):
        indices, y, var = self.get_sampled_dataset()        
        x = self.env.X[indices]
        self.gp.fit(x, y, var)
        
    def get_sampled_dataset(self):
        all_y = []
        all_var = []
        indices = []
        for i in range(self.env.num_samples):
            if len(self.camera_data[i])>0 and len(self.sensor_data[i])>0:
                yc = np.mean(self.camera_data[i])
                ys = np.mean(self.sensor_data[i])
                yeq = (self.camera_std**2 * ys + self.sensor_std**2 * yc) / (self.camera_std**2 + self.sensor_std**2)
                var = 1/(1/(self.sensor_std**2) + 1/(self.camera_std**2))

            elif len(self.sensor_data[i])>0:
                yeq = np.mean(self.sensor_data[i])
                var = self.sensor_std**2

            elif len(self.camera_data[i])>0:
                yeq = np.mean(self.camera_data[i])
                var = self.camera_std**2

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

    def _setup_ipp(self, criterion):
        self.criterion = criterion
        self.reset()
        self._post_update()

    def run_ipp(self, render=False, num_runs=10, criterion='entropy', camera_enabled=False, adaptive=False):
        # this function selects k points greedily and then finds the path that maximises the total information gain 

        assert criterion in ['entropy', 'monotonic_entropy', 'mutual_information'], 'Unknown criterion!!'
        self._setup_ipp(criterion)
        results = []
        for i in range(num_runs):
            print('\n==================================================================================================')
            print('Run {}/{}'.format(i+1, num_runs))
            run_start = time.time()
            
            # greedily select samples to be collected by sensors
            new_gp_indices = self.greedy(self.num_samples_per_batch)  
            waypoints = [tuple(self.env.gp_index_to_map_pose(x)) for x in new_gp_indices]
            next_sensor_locations = np.stack(waypoints)
            self.sensor_locations = np.concatenate([self.sensor_locations, next_sensor_locations]).astype(int)
            
            # Gather data along path only when camera is enabled
            if camera_enabled:          
                print('------ Finding valid paths ---------')
                print('Pose:',self.pose, 'Heading:', self.heading, 'Waypoints:', waypoints)
                start = time.time()
                paths_checkpoints, paths_indices, paths_cost = self.env.get_shortest_path_waypoints(self.pose, self.heading, waypoints, self.beta)
                end = time.time()
                print('Number of feasible paths: ', len(paths_indices))
                print('Time consumed {:.4f}'.format(end - start))

                print('\n------ Finding best path ----------')
                start = time.time()
                best_idx = self.best_path(paths_indices, new_gp_indices)
                next_path_indices = paths_indices[best_idx]
                next_path = np.stack(self.env.get_path_from_checkpoints(paths_checkpoints[best_idx]))
                end = time.time()
                print('Minimum cost: {} Best path cost: {}'.format(min(paths_cost), paths_cost[best_idx]))
                print('Time consumed {:.4f}'.format(end - start))
                
                if render:
                    self.env.render(paths_checkpoints[best_idx], self.path, next_sensor_locations, self.sensor_locations)
                
                # update agent statistics
                self.path = np.concatenate([self.path, next_path[1:]], axis=0).astype(int)
                self.pose = tuple(self.path[-1])
                self.heading = get_heading(self.path[-1], self.path[-2])
        
                # add samples (camera and sensor)
                self._add_samples(next_path_indices, source='camera')
            
            self._add_samples(new_gp_indices, source='sensor')
            
            print('\n-------- Prediction -------------- ')
            start = time.time()
            pred, cov = self.predict_test()
            var = np.diag(cov)
            rmse = compute_rmse(pred, self.env.test_Y)
            print('RMSE: {:.4f}'.format(rmse))
            print('Predictive Variance Max: {:.3f} Min: {:.3f} Mean: {:.3f}'.format(var.max(), var.min(), var.mean()))
            end = time.time()
            print('Time consumed {:.4f}'.format(end - start))

            # log predictions
            res = {'mean': pred, 'covariance': cov, 'rmse': rmse}
            results.append(res)

            # update GP model
            if adaptive and (i+1) % self.update_every == 0:
                print('\n---------- Updating model --------------')
                start = time.time()
                self.update_model()
                self._post_update()
                end = time.time()
                print('Time consumed {:.4f}'.format(end - start))

            run_end = time.time()
            print('\nTotal Time consumed in run {}: {:.4f}'.format(i+1, run_end - run_start))
            # ipdb.set_trace()

        print('==========================================================')
        print('--- Final statistics --- ')
        print('RMSE: {:.4f}'.format(rmse))
        print('Predictive Variance Max: {:.3f} Min: {:.3f} Mean: {:.3f}'.format(var.max(), var.min(), var.mean()))

        # TODO: log relevant things
        # path, camera samples, sensor samples
        # self.path, self.sensor_locations
        # save these statistics and path and sensing locations to compare results later on
        return results

    def predict_train(self, test_ind=None):
        test_ind = np.arange(self.env.num_samples) if test_ind is None else test_ind
        train_ind, train_y, train_var = self.get_sampled_dataset()

        cov_aa = self.cov_matrix[train_ind].T[train_ind].T + np.diag(train_var)
        cov_xx = self.cov_matrix[test_ind].T[test_ind].T
        cov_xa = self.cov_matrix[test_ind].T[train_ind].T

        mat1 = np.dot(cov_xa, np.linalg.inv(cov_aa))
        mu = np.dot(mat1, train_y.reshape(-1,1))
        cov = cov_xx - np.dot(mat1, cov_xa.T)
        return mu, np.diag(cov)

    def predict_test(self):
        train_ind, train_y, train_var = self.get_sampled_dataset()
        train_x = self.env.X[train_ind]
        mu, cov = posterior_distribution(self.gp, train_x, train_y, self.env.test_X, train_var, return_cov=True)
        return mu, cov

    def greedy(self, num_samples):
        # select most informative samples in a greedy manner
        n = self.env.num_samples
        camera_sampled = np.array([False if len(x)==0 else True for x in self.camera_data])
        camera_var = np.full(n, np.inf)
        camera_var[camera_sampled] = self.camera_std**2
        
        sensor_sampled = np.array([False if len(x)==0 else True for x in self.sensor_data])
        sensor_var = np.full(n, np.inf)
        sensor_var[sensor_sampled] = self.sensor_std**2
        
        sampled = sensor_sampled | camera_sampled
        var = np.minimum(sensor_var[sampled], camera_var[sampled])
        cov_v = self.cov_matrix[sampled].T[sampled].T + np.diag(var)
        ent_v = entropy_from_cov(cov_v, self.entropy_constant)

        cumm_utilities = []
        new_samples = []
        for _ in range(num_samples):
            utilities = np.full(n, 0.0)
            cond = ent_v + sum(cumm_utilities)

            for i in range(n):
                # if entropy is monotonic, then this step is not necessary (however better for efficiency)
                # NOTE: for camera sampled indices, the conditional entropy is negative so disallowing that also
                if sensor_sampled[i] or camera_sampled[i]:
                    continue

                # modify sampled (temporarily)
                sensor_sampled[i] = True
                sensor_var[i] = self.sensor_std**2
                sampled = sensor_sampled | camera_sampled
                var = np.minimum(sensor_var[sampled], camera_var[sampled])

                # a - set of all sampled locations 
                cov_a = self.cov_matrix[sampled].T[sampled].T + np.diag(var)
                ent_a = entropy_from_cov(cov_a, self.entropy_constant)
                if self.criterion == 'mutual_information':
                    cov_abar = self.cov_matrix[~sampled].T[~sampled].T 
                    ent_abar = entropy_from_cov(cov_abar)
                    cov_all = self.cov_matrix + np.diag(np.concatenate([var, np.zeros(sum(~sampled))]))
                    ent_all = entropy_from_cov(cov_all)
                    ut = ent_a + ent_abar - ent_all
                else:
                    ut = ent_a - cond

                utilities[i] = ut

                # reset sampled
                sensor_sampled[i] = False
                sensor_var[i] = np.inf

            best_sample = np.argmax(utilities)
            cumm_utilities.append(utilities[best_sample])
            new_samples.append(best_sample)

            # update sampled
            sensor_sampled[best_sample] = True
            sensor_var[best_sample] = self.sensor_std**2
 
            # if min(utilities) < 0:
            #     ipdb.set_trace()
        return new_samples

    def best_path(self, paths_indices, new_gp_indices):
        if len(paths_indices) == 1:
            return 0

        indices, _, var = self.get_sampled_dataset()
        all_ind = indices + new_gp_indices
        all_var = np.concatenate([var, np.full(len(new_gp_indices), self.sensor_std**2)])
        
        cov_aa = self.cov_matrix[all_ind].T[all_ind].T + np.diag(all_var)
        l_inv = np.linalg.inv(np.linalg.cholesky(cov_aa))
        cov_aa_inv = np.dot(l_inv.T, l_inv)

        all_ind_set = set(all_ind)
        all_h = []
        for i in range(len(paths_indices)):
            new_camera_ind = list(set(paths_indices[i]) - all_ind_set)
            cov_xx = self.cov_matrix[new_camera_ind].T[new_camera_ind].T
            cov_xa = self.cov_matrix[new_camera_ind].T[all_ind].T

            cov = cov_xx - np.dot(cov_xa, np.dot(cov_aa_inv, cov_xa.T))
            h = entropy_from_cov(cov, self.entropy_constant)
            all_h.append(h)
        idx = np.argmax(all_h)
        return idx

        