import numpy as np
import torch 
import time
from copy import deepcopy

from models import GPR
from graph_utils import get_heading
from utils import entropy_from_cov, compute_mae, predictive_distribution, find_shortest_path, find_equi_sample_path
# import ipdb


class Agent(object):
    def __init__(self, env, args, parent_agent=None, learn_likelihood_noise=True, mobile_std=None, static_std=None):
        super()
        self.env = env
        self.learn_likelihood_noise = learn_likelihood_noise
        self._init_model(args)

        self.static_std = args.static_std if static_std is None else static_std
        self.mobile_std = 10*self.static_std if mobile_std is None else mobile_std
        self.num_samples_per_batch = args.num_samples_per_batch
        self.update_every = args.update_every
        
        self.reset()
        if parent_agent is None:
            num_pretrain = int(args.fraction_pretrain * self.env.num_samples)
            self._pre_train(num_samples=num_pretrain)
        else:
            self.load_model(parent_agent)
            self.static_data = deepcopy(parent_agent.static_data)
            self.mobile_data = deepcopy(parent_agent.mobile_data)
            self.collected = deepcopy(parent_agent.collected)
            
    def _init_model(self, args):
        kernel_params = {'type': args.kernel}
        self.gp = GPR(latent=args.latent, lr=args.lr, max_iterations=args.max_iterations, kernel_params=kernel_params, learn_likelihood_noise=self.learn_likelihood_noise)

    def load_model(self, parent_agent):
        self.gp.reset(parent_agent.gp.train_x, parent_agent.gp.train_y, parent_agent.gp.train_var)
        self.gp.model.load_state_dict(parent_agent.gp.model.state_dict())
        
    def save_model(self, filename):
        state = {'state_dict': self.gp.model.state_dict()}
        torch.save(state, filename)

    def reset(self):
        self.pose = (0, 0)
        self.heading = (1, 0)
        self.path = np.copy(self.pose).reshape(-1, 2)
        self.collected = {'ind': [], 'std': [], 'y': []}
        self.static_locations = np.empty((0, 2))
        self.static_data = [[] for _ in range(self.env.num_samples)]
        self.mobile_data = [[] for _ in range(self.env.num_samples)]
        
    def _pre_train(self, num_samples):
        print('====================================================')
        print('--- Pretraining ---')
        self.pilot_survey(num_samples, self.static_std)
        self.update_model()
        
    def pilot_survey(self, num_samples, std):
        ind = np.random.permutation(self.env.num_samples)[:num_samples]
        self._add_samples(ind, stds=[std]*num_samples)
        
    def _add_samples(self, indices, stds):
        all_y = [None]*len(indices)
        for i in range(len(indices)):
            idx = indices[i]
            if idx == -1:
                continue
            y = self.env.collect_samples(idx, stds[i])
            all_y[i] = y
            if stds[i] == self.static_std:
                self.static_data[idx].append(y)
            else:
                self.mobile_data[idx].append(y)

        # update collected
        self.collected['ind'] += list(indices)
        self.collected['std'] += list(stds)
        self.collected['y'] += all_y

    def update_model(self):
        indices, y, var = self.get_sampled_dataset()        
        x = self.env.X[indices]
        self.gp.fit(x, y, var)
        
    def _post_update(self):
        self.cov_matrix = self.gp.cov_mat(x1=self.env.X, add_likelihood_var=True)
        
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

    def _setup_ipp(self, criterion, update):
        self.criterion = criterion
        # if not update:
        #     self.reset()
        self._post_update()

    def run_ipp(self, render=False, num_runs=10, criterion='entropy', mobile_enabled=True, update=False, slack=0, strategy='MaxEnt', disp=True):
        # informative path planner
        assert strategy in ['MaxEnt', 'Shortest', 'Equi-Sample'], 'Unknown strategy!!'
        assert criterion in ['entropy', 'mutual_information'], 'Unknown criterion!!'
        self._setup_ipp(criterion, update)

        test_error = []

        for i in range(num_runs):
            if disp:
                print('\n==================================================================================================')
                print('Run {}/{}'.format(i+1, num_runs))
            
            run_start = time.time()
            
            # greedily select static samples
            new_gp_indices = self.greedy(self.num_samples_per_batch)  
            waypoints = [tuple(self.env.gp_index_to_map_pose(x)) for x in new_gp_indices]
            next_static_locations = np.stack(waypoints)
            self.static_locations = np.concatenate([self.static_locations, next_static_locations]).astype(int)

            # Gather data along path 
            if mobile_enabled:    
                if disp:      
                    print('------ Finding valid paths ---------')
                    print('Pose:',self.pose, 'Heading:', self.heading, 'Waypoints:', waypoints)
                
                # find all paths 
                start = time.time()
                least_cost_ub = self.env.get_heuristic_cost(self.pose, self.heading, waypoints)
                if disp:
                    print('Least cost upper bound:',least_cost_ub)
                paths_checkpoints, paths_indices, paths_cost = self.env.get_all_paths(self.pose, self.heading, waypoints, least_cost_ub, slack)
                end = time.time()
                if disp:
                    print('Number of feasible paths: ', len(paths_indices))
                    print('Time consumed {:.4f}'.format(end - start))
                    print('\n------ Finding best path ----------')
                
                # find optimal path
                start = time.time()
                if strategy == 'Shortest':
                    best_idx = find_shortest_path(paths_cost)
                else:
                    best_idx = self.best_path(paths_indices, new_gp_indices)
                    if strategy == 'Equi-Sample':
                        best_idx = find_equi_sample_path(paths_indices, best_idx)
                end = time.time()

                if disp:
                    least_cost = min(paths_cost)
                    print('Least cost: {} Best path cost: {}'.format(least_cost, paths_cost[best_idx]))
                    print('Time consumed {:.4f}'.format(end - start))
                
                # update agent's record
                next_path = np.stack(self.env.get_path_from_checkpoints(paths_checkpoints[best_idx]))[1:]
                next_path_indices, stds = self.get_samples_sequence_from_path(next_path, waypoints)
                self.path = np.concatenate([self.path, next_path], axis=0).astype(int)
                self.pose = tuple(self.path[-1])
                self.heading = get_heading(self.path[-2], self.path[-1])
                
                if render:
                    pred = self.predict(self.env.all_x).reshape(self.env.shape)
                    true = self.env.all_y.reshape(self.env.shape)
                    self.env.render(paths_checkpoints[best_idx], self.path, next_static_locations, self.static_locations, true, pred)
                    # self.env.render(paths_checkpoints[best_idx], self.path, next_static_locations, self.static_locations)

            # TODO: a few things need to be done over here to make the else condition compatible 
            else:
                pass

            # gather samples
            self._add_samples(next_path_indices, stds)
            
            # update hyperparameters of GP model
            # TODO: this may not work properly right now
            if update and (i+1) % self.update_every == 0:
                if disp:
                    print('\n---------- Updating model --------------')
                start = time.time()
                self.update_model()
                self._post_update()
                end = time.time()
                if disp:
                    print('Time consumed {:.4f}'.format(end - start))

            # predict on test set
            if disp:
                print('\n-------- Prediction -------------- ')
            start = time.time()
            pred, var = self.predict(return_var=True)
            error = compute_mae(self.env.test_Y, pred)
            test_error.append(error)
            end = time.time()
            if disp:
                print('Test ERROR: {:.4f}'.format(error))
                print('Predictive Variance Max: {:.3f} Min: {:.3f} Mean: {:.3f}'.format(var.max(), var.min(), var.mean()))
                print('Time consumed {:.4f}'.format(end - start))

            run_end = time.time()
            if disp:
                print('\nTotal Time consumed in run {}: {:.4f}'.format(i+1, run_end - run_start))

        print('==========================================================')
        print('Strategy: {:s}'.format(strategy))
        print('--- Final statistics --- ')
        print('Test ERROR: {:.4f}'.format(error))
        print('Predictive Variance Max: {:.3f} Min: {:.3f} Mean: {:.3f}'.format(var.max(), var.min(), var.mean()))
        results = {'mean': pred, 'error':test_error}
        return results

    def predict(self, x=None, return_var=False, return_cov=False, return_mi=False):
        x = self.env.test_X if x is None else x
        train_ind, train_y, train_var = self.get_sampled_dataset()
        train_x = self.env.X[train_ind]
        return predictive_distribution(self.gp, train_x, train_y, x, train_var, return_var=return_var, return_cov=return_cov, return_mi=return_mi)

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
        ent_v = entropy_from_cov(cov_v)

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
                ent_a = entropy_from_cov(cov_a)
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
            ent_a = entropy_from_cov(cov_a)
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

    def run_naive(self, std, counts, metric='distance'):
        # traverse each row from start to end in a naive manner
        # counts should be list of ints
        # metric - either distance or sample

        test_error = []
        all_mi = []
        all_var = []

        for ns in counts:
            inds = []
            c = 0
            done = False
            while not done:
                # keep moving in the heading direction till you reach the end and need to shift to the next row
                next_pose = (self.pose[0]+self.heading[0], self.pose[1]+self.heading[1])
                ind = self.env.map_pose_to_gp_index_matrix[next_pose]
                if ind is not None:
                    inds.append(ind)

                if metric == 'samples':
                    if next_pose[0] == self.env.map.shape[0] - 1:
                        poses = [next_pose, (next_pose[0], next_pose[1]+1), (next_pose[0], next_pose[1]+2), (next_pose[0]-1, next_pose[1]+2)]
                        self.path = np.concatenate([self.path, poses], axis=0).astype(int)
                        self.heading = (-self.heading[0], 0)                       
                        self.pose = poses[-1]
                        
                    elif next_pose[0] == 0:
                        poses = [next_pose, (next_pose[0], next_pose[1]+1), (next_pose[0], next_pose[1]+2), (next_pose[0]+1, next_pose[1]+2)]
                        self.path = np.concatenate([self.path, poses], axis=0).astype(int)
                        self.heading = (-self.heading[0], 0)
                        self.pose = poses[-1]

                    else:
                        self.path = np.concatenate([self.path, [next_pose]], axis=0).astype(int)
                        self.pose = next_pose

                    done = len(inds)==ns

                elif metric == 'distance':
                    c += 1
                    self.path = np.concatenate([self.path, [next_pose]], axis=0).astype(int)
                    if next_pose[0]==0 and next_pose[1]%2==0:
                        self.heading = (0,1) if self.heading==(-1,0) else (1,0)
                    elif next_pose[0]==self.env.map.shape[0]-1 and next_pose[1]%2==0:
                        self.heading = (0,1) if self.heading==(1,0) else (-1,0)
                    self.pose = next_pose
                    done = c==ns
                else:
                    raise NotImplementedError

            self._add_samples(inds, [std]*len(inds))
            mu, cov, mi = self.predict(return_cov=True, return_mi=True)
            error = compute_mae(self.env.test_Y, mu)
            test_error.append(error)
            all_mi.append(mi)
            all_var.append(np.diag(cov).mean())

            # TODO: implement simulation rendering 

        var = np.diag(cov)
        strategy = 'Naive Static' if std==self.static_std else 'Naive Mobile'
        print('==========================================================')
        print('Strategy: ', strategy)
        print('--- Final statistics --- ')
        print('Test ERROR: {:.4f}'.format(error))
        print('Predictive Variance Max: {:.3f} Min: {:.3f} Mean: {:.3f}'.format(var.max(), var.min(), var.mean()))
        results = {'mean': mu, 'error': test_error, 'mi': all_mi, 'mean_var': all_var}
        return results

    def get_samples_sequence_from_path(self, path, waypoints):
        indices = []
        std = []
        sampled = [False]*len(waypoints)
        for loc in path:
            loc = tuple(loc)
            gp_index = self.env.map_pose_to_gp_index_matrix[loc]
            indices.append(gp_index if gp_index is not None else -1)
            if gp_index is not None:
                if loc in waypoints:
                    idx = waypoints.index(loc)
                    if not sampled[idx]:
                        std.append(self.static_std)
                        sampled[idx] = True
                    else:
                        std.append(self.mobile_std)
                else:
                    std.append(self.mobile_std)
            else:
                std.append(-1)
        return indices, std

    def prediction_vs_distance(self, k, num_runs):
        count = 0
        all_error = []
        all_mi = []
        all_var = []

        while count < k*num_runs:
            count += k
            inds = np.array(self.collected['ind'][:count])
            valid = inds!=-1

            x = self.env.X[inds[valid]]
            var = np.array(self.collected['std'])[:count][valid]**2
            y = np.array(self.collected['y'])[:count][valid]
            mu, cov, mi = predictive_distribution(self.gp, x, y, self.env.test_X, var, return_mi=True, return_cov=True)            

            error = compute_mae(self.env.test_Y, mu)
            all_error.append(error)
            all_mi.append(mi)
            all_var.append(np.diag(cov).mean())
        results = {'mean': mu, 'error': all_error, 'mi': all_mi, 'mean_var': all_var}
        return results