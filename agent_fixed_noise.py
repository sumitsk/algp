from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, RBF
import ipdb
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from utils import mi_change, conditional_entropy, is_valid_cell
from env import FieldEnv


class Agent(object):
    def __init__(self, env, model_type='GP'):
        super()
        self.model_type = model_type
        self.model = None
        
        self.camera_noise = 1.0
        self.sensor_noise = 0.05
        
        utility_types = ['entropy', 'information_gain']
        self.utility_type = utility_types[1]
        all_strategies = ['sensor_maximum_utility',
                          'camera_maximum_utility',
                          'informative']
        self.strategy = all_strategies[2]
        self._init_models()

        self.env = env
        feature_dim = 2
        # data collected by sensor and camera
        self.sensor_x = np.empty((0, feature_dim))
        self.sensor_y = np.empty((0,))
        self.camera_x = np.empty((0, feature_dim))
        self.camera_y = np.empty((0,))

        self.visited = np.zeros(env.shape)
        self.obs_y = np.zeros(env.shape)
        self.obs_var_inv = np.zeros(env.shape)

        self._pre_train(num_samples=20)
        # initial pose
        self.map_pose = (0, 0)
        self.search_radius = 10
        self.path = np.copy(self.map_pose).reshape(-1, 2)
        self.sensor_seq = np.empty((0, 2))

        self.model_update_every = 0
        self.last_update = 0

    def _init_models(self):
        if self.model_type == 'GP':
            # kernel = Matern(length_scale=1.0, nu=1.5)
            kernel = RBF(length_scale=1.0)
            self.model = gaussian_process.GaussianProcessRegressor(
                kernel)
        else:
            raise NotImplementedError

    def _pre_train(self, num_samples):
        ind = np.random.randint(0, self.env.num_samples, num_samples)
        self.add_samples(ind[:num_samples//2], self.camera_noise)
        self.add_samples(ind[num_samples//2:], self.sensor_noise)
        self.update_model()
        
    def add_samples(self, indices, noise):
        y = self.env.collect_samples(indices, noise)
        y_noise = np.full(len(indices), noise)
        self._handle_data(indices, y, y_noise)
        locations = self.env.gp_index_to_gp_pose(indices)
        self.visited[locations[:, 0], locations[:, 1]] = 1

    def _handle_data(self, indices, y, var):
        var_inv = 1.0/np.array(var)
        locations = self.env.gp_index_to_gp_pose(indices)
        old_obs = self.obs_y[locations[:, 0], locations[:, 1]]
        old_var_inv = self.obs_var_inv[locations[:, 0], locations[:, 1]]
        new_obs = (old_obs * old_var_inv + y * var_inv)/(old_var_inv + var_inv)
        new_var_inv = var_inv + old_var_inv

        self.obs_y[locations[:, 0], locations[:, 1]] = new_obs
        self.obs_var_inv[locations[:, 0], locations[:, 1]] = new_var_inv

    def update_model(self):
        indices = np.where(self.visited.flatten() == 1)[0]
        x = self.env.X[indices, :]
        var = 1.0/(self.obs_var_inv.flatten()[indices])
        y = self.obs_y.flatten()[indices]

        if self.model_type == 'GP':
            self.model.alpha = var
            self.model.fit(x, y)
        else:
            raise NotImplementedError

    def run(self, render=False, iterations=50):
        if render:
            plt.ion()

        if self.strategy == 'sensor_maximum_utility':
            raise NotImplementedError
            # self.step_max_utility('sensor', render, iterations)
        elif self.strategy == 'camera_maximum_utility':
            raise NotImplementedError
            # self.step_max_utility('camera', render, iterations)
        elif self.strategy == 'informative':
            self.step_informative(render, iterations)
        else:
            raise NotImplementedError

    def render(self):
        plt.figure(0)
        plt.set_cmap('hot')

        plot = 1.0 - np.repeat(self.env.map.oc_grid[:, :, np.newaxis], 3, axis=2)
        # highlight camera measurement
        if self.path.shape[0] > 0:
            plot[self.path[:, 0], self.path[:, 1], :] = [.75, .75, .5]
        # highlight sensor measurement
        if self.sensor_seq.shape[0] > 0:
            plot[self.sensor_seq[:, 0], self.sensor_seq[:, 1]] = [.05, 1, .05]

        plot[self.map_pose[0], self.map_pose[1], :] = [0, 0, 1]
        plt.imshow(plot, interpolation='nearest')
        plt.pause(.01)

    # def maximum_entropy(self, source):
    #     if source == 'sensor':
    #         model = self.sensor_model
    #     elif source == 'camera':
    #         model = self.camera_model
    #     else:
    #         raise NotImplementedError
    #     mu, std = model.predict(self.env.X, return_std=True)
    #     gp_indices = np.where(std == std.max())[0]
    #     map_poses = self.env.gp_index_to_map_pose(gp_indices)
    #     distances = self.env.map.get_distances(self.map_pose, map_poses)
    #     idx = np.argmin(distances)
    #     return gp_indices[idx], map_poses[idx]
    #
    # def step_max_utility(self, source, render, iterations):
    #     for i in range(iterations):
    #         next_gp_index, next_map_pose = self.maximum_entropy(source)
    #         self.add_samples(source, [next_gp_index])
    #         self.update_model(source)
    #         self._update_path(next_map_pose)
    #         self.map_pose = tuple(next_map_pose)
    #         if render:
    #             self.render()
    #         ipdb.set_trace()
    #
    # def _update_path(self, next_map_pose):
    #     pass
    #
    # def step_max_mi_change(self, source, render, iterations):
    #     for i in range(iterations):
    #         next_gp_index, next_map_pose = self.maximum_mi_change(source)
    #         self.add_samples(source, [next_gp_index])
    #         self.update_model(source)
    #         self._update_path(next_map_pose)
    #         self.map_pose = tuple(next_map_pose)
    #         if render:
    #             self.render()
    #         ipdb.set_trace()
    #
    # def maximum_mi_change(self, source):
    #     # computing change in mutual information is slow right now
    #     # Use entropy criteria for now
    #     if source == 'sensor':
    #         model = self.sensor_model
    #         mask = np.copy(self.sensor_visited.flatten())
    #     elif source == 'camera':
    #         model = self.camera_model
    #         mask = np.copy(self.camera_visited.flatten())
    #     else:
    #         raise NotImplementedError
    #     a_ind = np.where(mask == 1)[0]
    #     A = self.env.X[a_ind, :]
    #     a_bar_ind = np.where(mask == 0)[0]
    #     mi = np.zeros(self.env.num_samples)
    #     for i, x in enumerate(self.env.X):
    #         if mask[i] == 0:
    #             a_bar_ind = np.delete(a_bar_ind, np.where(a_bar_ind == i)[0][0])
    #         A_bar = self.env.X[a_bar_ind, :]
    #         info = mi_change(x, model, A, A_bar, model.alpha)
    #         mi[i] = info
    #
    #     gp_indices = np.where(mi == mi.max())[0]
    #     map_poses = self.env.gp_index_to_map_pose(gp_indices)
    #     distances = self.env.map.get_distances(self.map_pose, map_poses)
    #     idx = np.argmin(distances)
    #     return gp_indices[idx], map_poses[idx]

    def step_informative(self, render, iterations):
        for i in range(iterations):
            best_node = self._bfs_search(self.map_pose, self.search_radius)
            self._step(best_node)
            if render:
                self.render()
            # ipdb.set_trace()

    def _step(self, next_node):
        self.add_samples(next_node.parents_index, self.camera_noise)
        gp_index = self.env.map_pose_to_gp_index(next_node.map_pose)
        if gp_index is not None:
            self.add_samples([gp_index], self.sensor_noise)

        self.update_model()
        self.path = np.concatenate([self.path, next_node.path], axis=0).astype(int)
        self.map_pose = next_node.map_pose
        if self.env.map_pose_to_gp_index(self.map_pose) is not None:
            self.sensor_seq = np.concatenate([self.sensor_seq, np.array(self.map_pose).reshape(-1, 2)]).astype(int)

    def _bfs_search(self, map_pose, max_distance):
        node = Node(map_pose, 0, 0, [])
        open_nodes = [node]
        closed_nodes = []

        sz = self.env.map.oc_grid.shape
        gvals = np.ones(sz) * float('inf')
        gvals[map_pose] = 0
        cost = 1
        dx_dy = [(-1, 0), (1, 0), (0, 1), (0, -1)]
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
                if is_valid_cell(new_map_pose, sz) and \
                        self.env.map.oc_grid[new_map_pose] != 1 and \
                        new_gval < gvals[new_map_pose]:
                    gvals[new_map_pose] = new_gval

                    if gp_index is None:
                        new_parents_index = node.parents_index
                    else:
                        new_parents_index = node.parents_index + [gp_index]

                    new_path = np.concatenate([node.path, np.array(new_map_pose).reshape(-1, 2)])
                    new_node = Node(new_map_pose, new_gval, node.utility,
                                    new_parents_index, new_path)
                    open_nodes.append(new_node)

        # compute sensor and camera utility
        for node in closed_nodes:
            gp_index = self.env.map_pose_to_gp_index(node.map_pose)
            if gp_index is not None:
                indices = node.parents_index + [gp_index]
                x_noise = [self.camera_noise] * len(node.parents_index) + [self.sensor_noise]
            else:
                indices = node.parents_index
                x_noise = [self.camera_noise] * len(node.parents_index)
            # ipdb.set_trace()
            node.utility = self._get_utility(indices, x_noise)

        total_utility = [node.utility for node in closed_nodes]
        best_node = closed_nodes[np.argmax(total_utility).item()]
        return best_node

    # def plot_variance(self, source):
    #     if source == 'camera':
    #         mu, std = self.camera_model.predict(self.env.X, return_std=True)
    #         var = (std ** 2).reshape(self.env.shape)
    #     elif source == 'sensor':
    #         mu, std = self.sensor_model.predict(self.env.X, return_std=True)
    #         var = (std ** 2).reshape(self.env.shape)
    #     else:
    #         raise NotImplementedError
    #
    #     plt.title(source + ' variance plot')
    #     plt.imshow(var)
    #     plt.show()

    def _get_utility(self, indices, x_noise_var):
        x = self.env.X[indices, :]
        a = self.model.X_train_
        a_noise_var = self.model.alpha
        if self.utility_type == 'information_gain':
            unvisited_indices = np.where(self.visited.flatten() == 0)[0]
            a_bar = self.env.X[unvisited_indices, :]
            info = mi_change(x, a, a_bar, self.model.kernel,
                             x_noise_var, a_noise_var, a_bar_noise_var=None)
        elif self.utility_type == 'entropy':
            info = conditional_entropy(x, a, self.model.kernel,
                                       x_noise_var, a_noise_var)
        else:
            raise NotImplementedError
        ipdb.set_trace()
        return info


class Node(object):
    def __init__(self, map_pose, gval, utility, parents_index, path=np.empty((0, 2))):
        self.map_pose = map_pose
        self.gval = gval
        self.utility = utility
        self.parents_index = parents_index[:]
        self.path = np.copy(path)


if __name__ == '__main__':
    env = FieldEnv(num_rows=20, num_cols=20)
    agent = Agent(env, model_type='GP')
    agent.run(render=True)