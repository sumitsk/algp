import ipdb
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import entropy_from_cov, is_valid_cell, Node
from models import SklearnGPR, GpytorchGPR
import time
from methods import greedy


class Agent(object):
    def __init__(self, env, args):
        super()
        self.env = env
        self.gp = None
        self._init_models(args)

        self._camera_noise = args.camera_noise
        self._sensor_noise = args.sensor_noise
        self.utility_type = args.utility 
        self._strategy = args.strategy 
        # precision - reciprocal of variance
        self._precision_method = args.precision_method 

        self.reset()
        self._pre_train(num_samples=args.num_pretrain_samples, only_sensor=True)
        self.agent_map_pose = (0, 0)
        self.search_radius = args.search_radius
        self.search_radius_delta = args.search_radius_delta
        self.mi_radius = args.mi_radius

        self.path = np.copy(self.agent_map_pose).reshape(-1, 2)
        self.sensor_seq = np.empty((0, 2))
        self.stops = np.empty((0, 2))
        self.update_every = args.update_every

    def _init_models(self, args):
        if args.model_type == 'sklearn_GP':
            self.gp = SklearnGPR()
        elif args.model_type == 'gpytorch_GP':
            kernel_params = self._get_kernel_params(args)
            self.gp = GpytorchGPR(latent=args.latent, lr=args.lr, max_iterations=args.max_iterations, kernel_params=kernel_params)
        else:
            raise NotImplementedError

    def _get_kernel_params(self, args):
        kernel_params = {'type': args.kernel}
        if args.kernel == 'spectral_mixture':
            kernel_params['n_mixtures'] = args.n_mixtures
        return kernel_params

    def reset(self):
        self._visited = np.full(self.env.num_samples, False)
        self.obs_y = np.zeros(self.env.num_samples)
        self.obs_precision = np.zeros(self.env.num_samples)

    def _pre_train(self, num_samples, only_sensor=False):
        print('====================================================')
        print('--- Pretraining ---')
        ind = np.random.permutation(self.env.num_samples)[:num_samples]
        if only_sensor:
            self._add_samples(ind, self._sensor_noise)
        else:
            self._add_samples(ind[:num_samples//2], self._camera_noise)
            self._add_samples(ind[num_samples//2:], self._sensor_noise)
        self.update_model()
        # if don't want to remember/condition on pre_train data points    
        # self.reset()    
        
    def _add_samples(self, indices, noise):
        y = self.env.collect_samples(indices, noise)
        y_noise = np.full(len(indices), noise)
        self._handle_data(indices, y, y_noise)
        self._visited[indices] = True

    def _update_precision(self, old_precision, new_precision):
        if self._precision_method == 'sum':
            return old_precision + new_precision
        elif self._precision_method == 'max':
            return np.maximum(old_precision, new_precision)
        else:
            raise NotImplementedError

    def _handle_data(self, indices, y, var):
        precision = 1.0/np.array(var)
        old_obs = self.obs_y[indices]
        old_precision = self.obs_precision[indices]

        # weighted sum of old and new obs (y), weights are precision 
        new_obs = (old_obs * old_precision + y * precision)/(old_precision + precision)
        new_precision = self._update_precision(precision, old_precision)
        self.obs_y[indices] = new_obs
        self.obs_precision[indices] = new_precision

    def update_model(self):
        x = self.env.X[self._visited]
        var = 1.0/(self.obs_precision[self._visited])
        y = self.obs_y[self._visited]

        print('\n--- Updating GP model ---')
        start = time.time()
        self.gp.fit(x, y, var)
        end = time.time()
        print('Time consumed: {}'.format(end - start))

    def run(self, render=False, num_runs=40):
        if self._strategy == 'sensor_maximum_utility':
            raise NotImplementedError
        elif self._strategy == 'camera_maximum_utility':
            raise NotImplementedError
        elif self._strategy == 'informative':
            # final_pred, final_var = self.run_informative(render, num_runs)
            self.run2(num_runs=num_runs, render=render)
        else:
            raise NotImplementedError

        # final training and test rmse and variance
        # rmse_visited = np.linalg.norm(final_pred[self._visited] - self.env.Y[self._visited])/np.sqrt(self._visited.sum())
        # rmse_unvisited = np.linalg.norm(final_pred[~self._visited] - self.env.Y[~self._visited])/np.sqrt((~self._visited).sum())
        
        # print('==========================================================')
        # print('--- Final statistics --- ')
        # print('RMSE visited: {:.3f} unvisited: {:.3f}'.format(rmse_visited, rmse_unvisited))
        # print('Predictive Variance Max: {:.3f} Min: {:.3f} Mean: {:.3f}'.format(final_var.max(), final_var.min(), final_var.mean()))

        # TODO: log relevant things
        # path, camera samples, sensor samples
        # self.path, self.sensor_seq, self.stops
        # save these statistics and path and sensing locations to compare results later on

    # TODO: put render in map.py file
    def _render_path(self, ax):
        # ax.set_cmap('hot')
        plot = 1.0 - np.repeat(self.env.map.occupied[:, :, np.newaxis], 3, axis=2)
        # highlight camera measurement
        if self.path.shape[0] > 0:
            plot[self.path[:, 0], self.path[:, 1], :] = [.75, .75, .5]
        # highlight sensor measurement
        if self.sensor_seq.shape[0] > 0:
            plot[self.sensor_seq[:, 0], self.sensor_seq[:, 1]] = [.05, 1, .05]

        plot[self.agent_map_pose[0], self.agent_map_pose[1], :] = [0, 0, 1]
        ax.set_title('Environment')
        ax.imshow(plot)

    def render(self, fig, ax, pred, var):
        # TODO: improve render by drawing arrows and annotating
        # render path
        self._render_path(ax[0, 0])

        # TODO: use seaborn for rendering
        # render plots
        axt, axp, axv = ax[1, 0], ax[1, 1], ax[0, 1]
        axt.set_title('Ground Truth / Actual values')
        imt = axt.imshow(self.env.Y.reshape(self.env.shape),
            cmap='ocean', vmin=self.env.Y.min(), vmax=self.env.Y.max())
        div = make_axes_locatable(axt)
        caxt = div.new_horizontal(size='5%', pad=.05)
        fig.add_axes(caxt)
        fig.colorbar(imt, caxt, orientation='vertical')

        axp.set_title('Predicted values')
        imp = axp.imshow(pred.reshape(self.env.shape),
            cmap='ocean', vmin=self.env.Y.min(), vmax=self.env.Y.max())
        divm = make_axes_locatable(axp)
        caxp = divm.new_horizontal(size='5%', pad=.05)
        fig.add_axes(caxp)
        fig.colorbar(imp, caxp, orientation='vertical')

        axv.set_title('Variance')
        imv = axv.imshow(var.reshape(self.env.shape), cmap='hot')
        # divv = make_axes_locatable(axv)
        # caxv = divv.new_horizontal(size='5%', pad=.05)
        # fig.add_axes(caxv)
        # fig.colorbar(imv, caxv, orientation='vertical')

    def predict(self):
        pred, std = self.gp.predict(self.env.X, return_std=True)
        return pred, std**2

    def run_informative(self, render, num_runs):
        if render:
            plt.ion()
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        for i in range(num_runs):
            print('\n=============================================================')
            print('Run {}/{}'.format(i+1, num_runs))
            start = time.time()
            # find next node to visit (node contains the next path segment)
            next_node = self._bfs_search(self.agent_map_pose, self.search_radius + int(i * self.search_radius_delta))
            
            # add samples (camera and sensor)
            self._add_samples(next_node.parents_index, self._camera_noise)
            gp_index = self.env.map_pose_to_gp_index(next_node.map_pose)
            if gp_index is not None:
                self._add_samples([gp_index], self._sensor_noise)

            # update GP model
            if i % self.update_every == 0:
                self.update_model()

            # update agent statistics
            self.path = np.concatenate([self.path, next_node.path], axis=0).astype(int)
            self.agent_map_pose = next_node.map_pose
            if gp_index is not None:
                self.sensor_seq = np.concatenate([self.sensor_seq, np.array(self.agent_map_pose).reshape(-1, 2)]).astype(int)
            self.stops = np.concatenate([self.stops, np.array(self.agent_map_pose).reshape(-1, 2)]).astype(int)
    
            print('\n--- Prediction --- ')
            pred, var = self.predict()
            # TODO: max variance is often times quite low (check this)
            # TODO: implement a suitable terminating condition
            # if np.max(var) < .01:
            #     print('Converged')
            #     break
            print('Predictive Variance Max: {:.3f} Min: {:.3f} Mean: {:.3f}'.format(var.max(), var.min(), var.mean()))

            if render:
                self.render(fig, ax, pred, var)
                plt.pause(.1)
            end = time.time()
            print('\nTotal Time consumed in run {}: {:.4f}'.format(i+1, end - start))

        return pred, var
        
    # TODO: put path finding in map.py file
    def _bfs_search(self, map_pose, max_distance):
        print('Finding all possible paths')
        start = time.time()
        node = Node(map_pose, 0, 0, [])
        open_nodes = [node]
        closed_nodes = []

        sz = self.env.map.occupied.shape
        gvals = np.ones(sz) * float('inf')
        gvals[map_pose] = 0
        cost = 1
        dx_dy = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        
        # NOTE: right now there is only path for every cell (which may not be a big issue because of the grid layout of the field)
        # entropy/mi are not monotonic so you can't use a branch and bound method to terminate the search (verify this) 
        # TODO: if this is computationally expensive, then speed it up by using tree structure. 
        # the most expensive part is training GP model
        while len(open_nodes) != 0:
            node = open_nodes.pop(0)
            gp_index = self.env.map_pose_to_gp_index(node.map_pose)
            closed_nodes.append(deepcopy(node))
            map_pose = node.map_pose
            if node.gval >= max_distance:
                break

            for dx, dy in dx_dy:
                new_map_pose = (map_pose[0] + dx, map_pose[1] + dy)
                new_gval = node.gval + cost
                if is_valid_cell(new_map_pose, sz) and self.env.map.occupied[new_map_pose] != 1 and new_gval < gvals[new_map_pose]:
                    gvals[new_map_pose] = new_gval

                    # do not expand nodes in the opposite direction
                    if len(self.path) > 2 and node.map_pose == self.agent_map_pose and new_map_pose == tuple(self.path[-2]):
                        continue

                    if gp_index is None:
                        new_parents_index = node.parents_index
                    else:
                        new_parents_index = node.parents_index + [gp_index]

                    new_path = np.concatenate([node.path, np.array(new_map_pose).reshape(-1, 2)])
                    new_node = Node(new_map_pose, new_gval, node.utility, new_parents_index, new_path)
                    open_nodes.append(new_node)

        # NOTE: computational bottleneck (the most expensive part is GP training (reduce number of iterations there))
        all_nodes_indices = []
        all_nodes_x_noise = []
        for node in closed_nodes:
            gp_index = self.env.map_pose_to_gp_index(node.map_pose)
            if gp_index is not None:
                indices = node.parents_index + [gp_index]
                x_noise = [self._camera_noise] * len(node.parents_index) + [self._sensor_noise]
            else:
                indices = node.parents_index
                x_noise = [self._camera_noise] * len(node.parents_index)
            
            all_nodes_indices.append(indices)
            all_nodes_x_noise.append(x_noise)
        end = time.time()
        print('Time consumed: {:.4f}'.format(end - start))

        print('Finding best path')
        start = time.time()
        # v_ind - all gp indices in the mi_radius neighborhood
        if self.mi_radius > np.prod(self.env.map.shape):
            v_ind = list(range(self.env.num_samples))
        else:
            v_ind = self.env.get_neighborhood(self.agent_map_pose, self.mi_radius)[1]
 
        # samples outside mi_radius region (will be ignored for computing MI)
        rest = list(set(np.arange(self.env.num_samples)) - set(v_ind))

        v = self.env.X[v_ind, :]
        cov_v = self.gp.cov_mat(v, v, white_noise=None)
        info = []

        # a - all samples taken so far (with their data dependent noise)
        # a* = argmax MI(a, a_bar) where a_bar = v \ a
        # compute MI(a, a_bar) = H(a) + H(a_bar) - H(a, a_bar) for all a
        for indices, noise in zip(all_nodes_indices, all_nodes_x_noise):
            if len(indices) == 0:
                info.append(-np.inf)
                continue

            precision = np.copy(self.obs_precision)
            # update samples' precision (reciprocal of variance) in the current path (a)
            precision[indices] = self._update_precision(precision[indices], 1.0 / np.array(noise))
            precision_v = precision[v_ind]

            vis = np.copy(self._visited)
            vis[indices] = True
            vis[rest] = False

            cov_a = cov_v[vis[v_ind]].T[vis[v_ind]].T
            a_noise = 1.0/precision_v[vis[v_ind]]
        
            # entropy of new A
            e1 = entropy_from_cov(cov_a + np.diag(a_noise))
            if self.utility_type == 'entropy':
                info.append(e1)
            else:
                vis[rest] = True
                unvis = ~vis
                # precision_v[unvis[v_ind]] = self._camera_noise
                precision_v[unvis[v_ind]] = np.inf
                cov_a_bar = cov_v[unvis[v_ind]].T[unvis[v_ind]].T
                a_bar_noise = 1.0/precision_v[unvis[v_ind]]

                # entropy of new A_bar (V - A)
                e2 = entropy_from_cov(cov_a_bar + np.diag(a_bar_noise))
                v_noise = 1.0/precision_v
                e3 = entropy_from_cov(cov_v + np.diag(v_noise))

                info.append(e1 + e2 - e3)

        best_node = closed_nodes[np.argmax(info).item()]
        end = time.time()
        print('Time consumed: {:.4f}'.format(end - start))
        return best_node

    # TODO: make a separate file which implements strategies like greedy, etc.
    # this function selects k points greedily and then finds a path that maximises the total information gain 
    def run2(self, num_runs, k=1, render=False):
        if render:
            plt.ion()
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))
            ipdb.set_trace()

        # k - number of samples per run
        for i in range(num_runs):

            # select k samples greedily (all of which will be collected by sensor)
            cov = self.gp.cov_mat(self.env.X)
            new_samples, utilities = greedy(self.env.X, self._visited, cov, k, pose=self.agent_map_pose, locs=self.env.locs, max_distance=8)
            # ipdb.set_trace()

            new_locs = self.env.gp_index_to_map_pose(new_samples)
            new_locs = [tuple(x) for x in new_locs]
            # new_samples are the gp indices here
            # convert them to map poses and select a goal location
            print('------ Finding paths ---------')
            print('waypoints ', new_locs)
            start = time.time()
            if i==0:
                delta = None
            else:
                delta = self.path[-1][0] - self.path[-2][0]
            paths = self.env.map.all_paths(self.agent_map_pose, new_locs, delta)
            
            # ipdb.set_trace()
            # convert map poses to gp index
            all_utilities = []
            all_indices = []

            for path in paths:
                indices = []
                for loc in path:
                    idx = self.env.map_pose_to_gp_index(tuple(loc))
                    if idx is not None:
                        indices.append(idx)
                all_indices.append(indices)
                temp_cov = cov[indices].T[indices].T
                var = np.eye(len(indices))*self._camera_noise
                var[-1,-1] = self._sensor_noise
                ut = entropy_from_cov(temp_cov + var)
                all_utilities.append(ut)        

            if len(paths) == 0:
                ipdb.set_trace()
                paths = self.env.map.all_paths(self.agent_map_pose, new_locs, delta)
                            
            idx = np.argmax(all_utilities)

            best_indices = all_indices[idx]
            best_path = paths[idx]
            # add samples (camera and sensor)
            self._add_samples(best_indices[:-1], self._camera_noise)
            self._add_samples([best_indices[-1]], self._sensor_noise)

            # update GP model
            if i % self.update_every == 0:
                self.update_model()

            # update agent statistics
            self.path = np.concatenate([self.path, best_path[1:]], axis=0).astype(int)
            self.agent_map_pose = tuple(self.path[-1])

            self.sensor_seq = np.concatenate([self.sensor_seq, np.stack(new_locs)]).astype(int)
            self.stops = np.concatenate([self.stops, np.stack(new_locs)]).astype(int)
    
            print('\n--- Prediction --- ')
            pred, var = self.predict()
            # TODO: max variance is often times quite low (check this)
            # TODO: implement a suitable terminating condition
            # if np.max(var) < .01:
            #     print('Converged')
            #     break
            print('Predictive Variance Max: {:.3f} Min: {:.3f} Mean: {:.3f}'.format(var.max(), var.min(), var.mean()))

            if render:
                self.render(fig, ax, pred, var)
                plt.pause(.1)
            end = time.time()
            print('\nTotal Time consumed in run {}: {:.4f}'.format(i+1, end - start))

        return pred, var
        