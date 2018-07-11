from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, RBF
import ipdb
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from utils import mi_change, conditional_entropy
from env import FieldEnv


class Agent(object):
    def __init__(self, env, sensor_model_type='GP', camera_model_type='GP'):
        super()
        self.sensor_model_type = sensor_model_type
        self.camera_model_type = camera_model_type
        self.sensor_model = None
        self.camera_model = None

        self.camera_noise = 1.0
        self.sensor_noise = 0.05
        # total utility = sensor_utility + alpha * camera_utility

        # todo: clean this up
        self.alpha = .5
        utility_types = ['entropy', 'information_gain']
        self.sensor_utility_type = utility_types[1]
        self.camera_utility_type = utility_types[1]
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

        # sensor measurements have been taken from locations set to 1
        self.sensor_visited = np.zeros(env.shape)
        # camera measurements have been taken from locations set to 1
        self.camera_visited = np.zeros(env.shape)

        self._pre_train(num_samples=20)
        # initial pose
        self.map_pose = (0, 0)
        self.search_radius = 10
        self.path = np.copy(self.map_pose).reshape(-1, 2)
        self.sensor_seq = np.empty((0, 2))

        self.sensor_model_update_every = 0
        self.camera_model_update_every = 0
        self.last_sensor_update = 0
        self.last_camera_update = 0

    @property
    def camera_samples_count(self):
        return self.camera_x.shape[0]

    @property
    def sensor_samples_count(self):
        return self.sensor_x.shape[0]

    def _init_models(self):
        if self.sensor_model_type == 'GP':
            sensor_kernel = Matern(length_scale=1.0, nu=1.5)
            # sensor_kernel = RBF(length_scale=1.0)
            self.sensor_model = gaussian_process.GaussianProcessRegressor(
                sensor_kernel, self.sensor_noise ** 2)
        else:
            raise NotImplementedError

        if self.camera_model_type == 'GP':
            camera_kernel = Matern(length_scale=1.0, nu=1.5)
            # camera_kernel = RBF(length_scale=1.0)
            self.camera_model = gaussian_process.GaussianProcessRegressor(
                camera_kernel, self.camera_noise ** 2)
        else:
            raise NotImplementedError

    def _pre_train(self, num_samples):
        ind = np.random.randint(0, self.env.X.shape[0], num_samples)
        self.add_samples('sensor', ind)
        self.add_samples('camera', ind)
        self.update_model('sensor')
        self.update_model('camera')

    def add_samples(self, source, indices):
        if source == 'camera':
            tmp = self.camera_visited.flatten()
            noise = self.camera_noise
        elif source == 'sensor':
            tmp = self.sensor_visited.flatten()
            noise = self.sensor_noise
        else:
            raise NotImplementedError

        # filter out locations already sampled
        # this is important because for same X we can't have two different Y
        remove = np.where(tmp[indices] == 1)[0]
        final_indices = np.delete(indices, remove, axis=0)
        y = self.env.collect_samples(final_indices, noise)
        x = self.env.X[final_indices, :]
        tmp[final_indices] = 1

        if source == 'sensor':
            self.sensor_x = np.concatenate([self.sensor_x, x])
            self.sensor_y = np.concatenate([self.sensor_y, y])
            self.sensor_visited = tmp.reshape(self.env.shape)
        else:
            self.camera_x = np.concatenate([self.camera_x, x])
            self.camera_y = np.concatenate([self.camera_y, y])
            self.camera_visited = tmp.reshape(self.env.shape)

    def update_model(self, source):
        # print(source, ' updated')
        if source == 'sensor':
            if self.sensor_model_type == 'GP':
                self.sensor_model.fit(self.sensor_x, self.sensor_y)
                self.last_sensor_update = self.sensor_samples_count
        elif source == 'camera':
            if self.camera_model_type == 'GP':
                self.camera_model.fit(self.camera_x, self.camera_y)
                self.last_camera_update = self.camera_samples_count
        else:
            raise NotImplementedError

    def run(self, render=False, iterations=50):
        if render:
            plt.ion()

        if self.strategy == 'sensor_maximum_utility':
            self.step_max_utility('sensor', render, iterations)
        elif self.strategy == 'camera_maximum_utility':
            self.step_max_utility('camera', render, iterations)
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

    def maximum_entropy(self, source):
        if source == 'sensor':
            model = self.sensor_model
        elif source == 'camera':
            model = self.camera_model
        else:
            raise NotImplementedError
        mu, std = model.predict(self.env.X, return_std=True)
        gp_indices = np.where(std == std.max())[0]
        map_poses = self.env.gp_index_to_map_pose(gp_indices)
        distances = self.env.map.get_distances(self.map_pose, map_poses)
        idx = np.argmin(distances)
        return gp_indices[idx], map_poses[idx]

    def step_max_utility(self, source, render, iterations):
        for i in range(iterations):
            next_gp_index, next_map_pose = self.maximum_entropy(source)
            self.add_samples(source, [next_gp_index])
            self.update_model(source)
            self._update_path(next_map_pose)
            self.map_pose = tuple(next_map_pose)
            if render:
                self.render()
            ipdb.set_trace()

    def _update_path(self, next_map_pose):
        pass

    def step_max_mi_change(self, source, render, iterations):
        for i in range(iterations):
            next_gp_index, next_map_pose = self.maximum_mi_change(source)
            self.add_samples(source, [next_gp_index])
            self.update_model(source)
            self._update_path(next_map_pose)
            self.map_pose = tuple(next_map_pose)
            if render:
                self.render()
            ipdb.set_trace()

    def maximum_mi_change(self, source):
        # computing change in mutual information is slow right now
        # Use entropy criteria for now
        if source == 'sensor':
            model = self.sensor_model
            mask = np.copy(self.sensor_visited.flatten())
        elif source == 'camera':
            model = self.camera_model
            mask = np.copy(self.camera_visited.flatten())
        else:
            raise NotImplementedError
        a_ind = np.where(mask == 1)[0]
        A = self.env.X[a_ind, :]
        a_bar_ind = np.where(mask == 0)[0]
        mi = np.zeros(self.env.X.shape[0])
        for i, x in enumerate(self.env.X):
            if mask[i] == 0:
                a_bar_ind = np.delete(a_bar_ind, np.where(a_bar_ind == i)[0][0])
            A_bar = self.env.X[a_bar_ind, :]
            info = mi_change(x, model, A, A_bar, model.alpha)
            mi[i] = info

        gp_indices = np.where(mi == mi.max())[0]
        map_poses = self.env.gp_index_to_map_pose(gp_indices)
        distances = self.env.map.get_distances(self.map_pose, map_poses)
        idx = np.argmin(distances)
        return gp_indices[idx], map_poses[idx]

    def step_informative(self, render, iterations):
        for i in range(iterations):
            best_node = self._bfs_search(self.map_pose, self.search_radius)
            self._step(best_node)
            if render:
                self.render()
            # ipdb.set_trace()

    def _step(self, next_node):
        gp_index = self.env.map_pose_to_gp_index(next_node.map_pose)
        if gp_index is not None:
            indices = next_node.parents_index + [gp_index]
            self.add_samples('sensor', [gp_index])
        else:
            indices = next_node.parents_index
        self.add_samples('camera', indices)
        self.update_model('camera')
        self.update_model('sensor')

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

            # don't compute utility here
            # camera_utility = self._get_utility('camera', gp_index, node.parents_index)
            # sensor_utility = self._get_utility('sensor', gp_index, node.parents_index)
            camera_utility = 0
            sensor_utility = 0
            # add sensor utility (a closed node now)
            node_copy = deepcopy(node)
            node_copy.sensor_utility = sensor_utility
            closed_nodes.append(node_copy)
            # add camera utility to the current node
            node.camera_utility += camera_utility
            map_pose = node.map_pose
            if node.gval >= max_distance:
                break

            for dx, dy in dx_dy:
                new_map_pose = (map_pose[0] + dx, map_pose[1] + dy)
                new_gval = node.gval + cost
                if isvalid(new_map_pose, sz) and self.env.map.oc_grid[new_map_pose] != 1 and \
                        new_gval < gvals[new_map_pose]:
                    gvals[new_map_pose] = new_gval

                    if gp_index is None:
                        new_parents_index = node.parents_index
                    else:
                        new_parents_index = node.parents_index + [gp_index]

                    new_path = np.concatenate([node.path, np.array(new_map_pose).reshape(-1, 2)])
                    new_node = Node(new_map_pose, new_gval, node.camera_utility,
                                    new_parents_index, new_path)
                    open_nodes.append(new_node)

        # compute sensor and camera utility
        inf = float('inf')
        utilities = inf * np.ones(self.env.num_samples)
        for node in closed_nodes:
            gp_index = self.env.map_pose_to_gp_index(node.map_pose)
            if gp_index is not None:
                if utilities[gp_index] == inf:
                    sensor_util = self._get_utility('sensor', gp_index)
                    node.sensor_utility = sensor_util
                    utilities[gp_index] = sensor_util
                else:
                    node.sensor_utility = utilities[gp_index]

                indices = node.parents_index + [gp_index]
            else:
                indices = node.parents_index
            camera_util = self._get_utility('camera', indices)
            node.camera_utility = camera_util

        total_utility = [self.alpha * node.camera_utility + node.sensor_utility for node in closed_nodes]
        best_node = closed_nodes[np.argmax(total_utility).item()]
        return best_node

    def plot_variance(self, source):
        if source == 'camera':
            mu, std = self.camera_model.predict(self.env.X, return_std=True)
            var = (std ** 2).reshape(self.env.shape)
        elif source == 'sensor':
            mu, std = self.sensor_model.predict(self.env.X, return_std=True)
            var = (std ** 2).reshape(self.env.shape)
        else:
            raise NotImplementedError

        plt.title(source + ' variance plot')
        plt.imshow(var)
        plt.show()

    def _get_utility(self, source, indices):
        if source == 'camera':
            model = self.camera_model
            util = self.camera_utility_type
            mask = np.copy(self.camera_visited.flatten())
        elif source == 'sensor':
            model = self.sensor_model
            util = self.sensor_utility_type
            mask = np.copy(self.sensor_visited.flatten())
        else:
            raise NotImplementedError

        x = self.env.X[indices, :]
        A = model.X_train_
        if util == 'information_gain':
            mask[indices] = 1
            unvisited_indices = np.where(mask == 0)[0]
            A_bar = self.env.X[unvisited_indices, :]
            return mi_change(x, A, A_bar, model.kernel, model.alpha)
        elif util == 'entropy':
            return conditional_entropy(x, A, model.kernel, model.alpha)
        else:
            raise NotImplementedError


def isvalid(node, shape):
    if 0 <= node[0] < shape[0] and 0 <= node[1] < shape[1]:
        return True
    return False


class Node(object):
    def __init__(self, map_pose, gval, utility, parents_index, path=np.empty((0, 2))):
        self.map_pose = map_pose
        self.gval = gval
        self.camera_utility = utility
        self.sensor_utility = 0
        self.parents_index = parents_index[:]
        self.path = np.copy(path)


if __name__ == '__main__':
    env = FieldEnv(num_rows=20, num_cols=20)
    agent = Agent(env, sensor_model_type='GP', camera_model_type='GP')
    agent.run(render=True)