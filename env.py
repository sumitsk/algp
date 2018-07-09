from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, RBF
import ipdb
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt 

from map import Map
from utils import generate_gaussian_data, conditional_entropy, mutual_information


class Agent(object):
    def __init__(self, env, sensor_model_type='GP', camera_model_type='GP'):
        super()
        self.sensor_model_type = sensor_model_type
        self.camera_model_type = camera_model_type
        self.sensor_model = None
        self.camera_model = None
        self.sensor_model_update_every = 1
        self.camera_model_update_every = 1
        self.last_sensor_update = 0
        self.last_camera_update = 0
        self.camera_noise = 1.0
        self.sensor_noise = 0.05
        self.sensor_utility_type = 'mutual information'
        self.camera_utility_type = 'mutual information'
        self._init_models()

        self.env = env
        feature_dim = 2
        # data collected by sensor and camera
        self.sensor_x = np.empty((0, feature_dim))
        self.sensor_y = np.empty((0,))
        self.camera_x = np.empty((0, feature_dim))
        self.camera_y = np.empty((0,))

        # initial pose
        self.map_pose = (0, 0)
        self.search_radius = 40

        # sensor measurements have been taken from locations set to 1
        self.sensor_visited = np.zeros(env.shape)
        # camera measurements have been taken from locations set to 1
        self.camera_visited = np.zeros(env.shape)

        self._pre_train(num_samples=10)
        self.path = np.empty((0, 2))

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
                sensor_kernel, self.sensor_noise**2)
        else:
            raise NotImplementedError

        if self.camera_model_type == 'GP':
            camera_kernel = Matern(length_scale=1.0, nu=1.5)
            # camera_kernel = RBF(length_scale=1.0)
            self.camera_model = gaussian_process.GaussianProcessRegressor(
                camera_kernel, self.camera_noise**2)
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

    def run(self, render=False):
        if render:
            plt.ion()

        for i in range(50):
            print('Run ', i+1)
            nodes = self.bfs_search(self.map_pose, self.search_radius)
            best_node = self.select_next_node(nodes)
            self.step(best_node, render)

    def step(self, node, render=False):
        # add camera samples
        if len(node.parents_index) > 0:
            self.add_samples('camera', node.parents_index)
        camera_diff = self.camera_samples_count - self.last_camera_update
        if camera_diff >= self.camera_model_update_every:
            self.update_model('camera')

        self.path = np.concatenate([self.path, node.path]).astype(int)
        self.map_pose = node.map_pose

        gp_index = self._map_pose_to_gp_index(node.map_pose)
        if gp_index is not None:
            # adding camera measurement also at the sensor sampling space
            self.add_samples('camera', [gp_index])
            self.add_samples('sensor', [gp_index])
        sensor_diff = self.sensor_samples_count - self.last_sensor_update
        if sensor_diff >= self.sensor_model_update_every:
            self.update_model('sensor')

        if render:
            self.render()
        print(self.path)

    def render(self):
        # plt.figure(0)
        plot = self.env.map.oc_grid

        # highlight camera measurement
        plot[self.path[:, 0], self.path[:, 1]] = 0.5
        plot[self.map_pose] = 0.7
        plt.imshow(plot)
        plt.pause(.1)
        # plt.show()

    # @deprecated
    # def select(self, map_pose, max_distance):
    #     # NOTE: this is a test I ran to validate the computation of covariance matrix via in-built function and
    #     # via actual formula (first principle lets say), see result below
    #
    #     sensor_unvisited = np.where(self.sensor_visited.flatten() == 0)[0]
    #     # sensor_visited = np.where(self.sensor_visited.flatten() == 1)[0]
    #     sensor_Y = self.env.X[sensor_unvisited, :]
    #     sensor_A = self.sensor_x
    #
    #     sensor_KYA = self.sensor_model.kernel_(sensor_Y, sensor_A)
    #     sensor_KAA = self.sensor_model.kernel_(sensor_A, sensor_A)
    #     sensor_KYY = self.sensor_model.kernel_(sensor_Y, sensor_Y)
    #
    #     # adding sensor_noise to the diagonal elements of KAA satisfies eq2
    #     # Possible bug: sometimes off-diagonal elements are not close by
    #     sensor_KAA += self.sensor_noise * np.eye(sensor_KAA.shape[0])
    #
    #     camera_unvisited = np.where(self.camera_visited.flatten() == 0)[0]
    #     # camera_visited = np.where(self.camera_visited.flatten() == 1)[0]
    #     camera_Y = self.env.X[camera_unvisited, :]
    #     camera_A = self.camera_x
    #
    #     camera_KYA = self.sensor_model.kernel_(camera_Y, camera_A)
    #     camera_KAA = self.sensor_model.kernel_(camera_A, camera_A)
    #     camera_KYY = self.sensor_model.kernel_(camera_Y, camera_Y)
    #
    #     # Method1: select location which leads to maximum reduction in entropy
    #     mu1, std = self.sensor_model.predict(sensor_Y, return_std=True)
    #     mu2, sigma = self.sensor_model.predict(sensor_Y, return_cov=True)
    #
    #     # verify eq(2) for a vector of inputs
    #     # mat1 = sensor_KYY - np.dot(np.dot(sensor_KYA, np.linalg.inv(sensor_KAA)), sensor_KYA.T)
    #
    #     # RESULT: mat1 is almost always equal to sigma. Sometimes off-diagonal terms are quite off
    #
    #     # err = np.abs(mat1 - sigma).max()
    #     # print(err)
    #     # if err > .01:
    #     #     ipdb.set_trace()

    # @deprecated
    # def nearby_locations(self, pose, max_distance):
    #     # returns all locations which are less than max distance apart from the current pose
    #     # run a BFS search and find all unvisited locations {O(max_distance)}
    #
    #     locations = self.env.map.get_nearby_locations(pose, max_distance)
    #
    #     # remove those which have already been sensed
    #     ind = np.ravel_multi_index(locations.T, self.sensor_visited.shape)
    #     remove_indices = np.where(self.sensor_visited.flatten()[ind] == 1)[0]
    #     locations = np.delete(locations, remove_indices, axis=0)
    #     return locations

    def bfs_search(self, map_pose, max_distance):
        node = Node(map_pose, 0, 0, [])
        open_nodes = [node]
        closed_nodes = []

        sz = self.env.map.oc_grid.shape
        gvals = np.ones(sz) * float('inf')
        gvals[map_pose] = 0
        cost = 1
        dxdy = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        while len(open_nodes) != 0:
            node = open_nodes.pop(0)
            # print(node.map_pose)
            gp_index = self._map_pose_to_gp_index(node.map_pose)

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

            for dx, dy in dxdy:
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
        return closed_nodes

    def select_next_node(self, closed_nodes):
        # closed_nodes is now a list of potential next sampling locations
        # select the most informative node now

        # compute sensor utility
        utilities = -1 * np.ones(self.env.X.shape[0])
        for node in closed_nodes:
            gp_index = self._map_pose_to_gp_index(node.map_pose)
            if gp_index is not None:
                if utilities[gp_index] == -1:
                    sensor_util = self._get_utility('sensor', gp_index)
                    node.sensor_utility = sensor_util
                    utilities[gp_index] = sensor_util
                else:
                    node.sensor_utility = utilities[gp_index]

        su = [node.sensor_utility for node in closed_nodes]

        # compute camera utility
        for node in closed_nodes:
            gp_index = self._map_pose_to_gp_index(node.map_pose)
            if gp_index is not None:
                indices = node.parents_index + [gp_index]
            else:
                indices = node.parents_index
            camera_util = self._get_utility('camera', indices)
            node.camera_utility = camera_util

        cu = [node.camera_utility for node in closed_nodes]

        # TODO: how about a weighted sum?
        total_utility = [node.camera_utility + node.sensor_utility for node in closed_nodes]

        ipdb.set_trace()
        # divide by distance
        # total_utility = [(node.camera_utility + node.sensor_utility)/node.gval for node in closed_nodes]
        # subtract distance
        # total_utility = [(node.camera_utility + node.sensor_utility) - node.gval for node in closed_nodes]

        best_node = closed_nodes[np.argmax(total_utility)]
        return best_node

    def plot_variance(self, source):
        if source == 'camera':
            mu, std = self.camera_model.predict(self.env.X, return_std=True)
            var = (std**2).reshape(self.env.shape)
        elif source == 'sensor':
            mu, std = self.sensor_model.predict(self.env.X, return_std=True)
            var = (std**2).reshape(self.env.shape)
        else:
            raise NotImplementedError

        plt.title(source + ' variance plot')
        plt.imshow(var)
        plt.show()

    def _gp_index_to_map_pose(self, gp_index):
        gp_pose = self._gp_index_to_gp_pose(gp_index)
        map_pose = self.env.map.gp_pose_to_map_pose(gp_pose)
        return map_pose
                    
    def _map_pose_to_gp_index(self, map_pose):
        # if isinstance(map_pose, tuple):
        #     tmp = np.array(map_pose)
        # else:
        #     tmp = map_pose
        assert isinstance(map_pose, tuple), 'Map pose should be a tuple'
        gp_pose = self.env.map.map_pose_to_gp_pose(map_pose)
        if gp_pose is None:
            return None
        return self._gp_pose_to_gp_index(gp_pose)

    def _gp_pose_to_gp_index(self, gp_pose):
        if isinstance(gp_pose, tuple):
            tmp = np.array(gp_pose)
        else:
            tmp = np.copy(gp_pose)
        # print(gp_pose, tmp)
        indices = np.ravel_multi_index(tmp.reshape(-1, 2).T, self.env.shape)
        if isinstance(gp_pose, tuple):
            return indices[0]
        return indices

    def _gp_index_to_gp_pose(self, gp_index):
        return np.vstack(np.unravel_index(gp_index, self.env.shape)).T

    # @deprecated
    # def _get_utility(self, source, gp_index, parents_index):
    #     # TODO: this may be inaccurate
    #     # return 0 for free cell (which doesn't have any samples)
    #     if gp_index is None:
    #         return 0
    #
    #     if source == 'camera':
    #         model = self.camera_model
    #         kernel = self.camera_model.kernel_
    #         visited = self.camera_visited
    #         noise = self.camera_noise
    #         utility = self.camera_utility_type
    #
    #     elif source == 'sensor':
    #         model = self.sensor_model
    #         kernel = self.sensor_model.kernel_
    #         visited = self.sensor_visited
    #         noise = self.sensor_noise
    #         utility = self.sensor_utility_type
    #     else:
    #         raise NotImplementedError
    #
    #     temp = np.copy(visited.flatten())
    #     # TODO: is this the right thing to do? (possibly a bug: entropy is negative at times)
    #     # no information if already visited
    #     if temp[gp_index] == 1:
    #         return 0
    #
    #     # add all parent_nodes in visited list
    #     if len(parents_index) > 0:
    #         temp[parents_index] = 1
    #
    #     # to exclude node from both lists
    #     temp[gp_index] = 2
    #     visited_indices = np.where(temp == 1)[0]
    #     unvisited_indices = np.where(temp == 0)[0]
    #
    #     y = self.env.X[gp_index, :]
    #     A = self.env.X[visited_indices, :]
    #     A_bar = self.env.X[unvisited_indices, :]
    #
    #     e1 = entropy(y, A, kernel, noise)
    #     ipdb.set_trace()
    #     if utility == 'entropy':
    #         return e1
    #     elif utility == 'information_gain':
    #         e2 = entropy(y, A_bar, kernel, noise)
    #         info = e1 - e2
    #         return info
    #     else:
    #         raise NotImplementedError

    def _get_utility(self, source, indices):
        if source == 'camera':
            model = self.camera_model
            util = self.camera_utility_type
            noise_std = self.camera_noise
        elif source == 'sensor':
            model = self.sensor_model
            util = self.sensor_utility_type
            noise_std = self.sensor_noise
        else:
            raise NotImplementedError

        x = self.env.X[indices, :]
        if util == 'mutual information':
            return mutual_information(x, model, noise_std)
        elif util == 'entropy':
            return conditional_entropy(x, model)
        else:
            raise NotImplementedError


def entropy(y, A, kernel, noise):
    y = y.reshape(-1, 2)
    A = A.reshape(-1, 2)
    sigma_AA = kernel(A, A) + noise * np.eye(A.shape[0])
    sigma_yA = kernel(y, A)

    var = kernel(y, y) - np.dot(np.dot(sigma_yA, np.linalg.inv(sigma_AA)), sigma_yA.T)
    var = var[0, 0]
    entropy = .5*np.log(2*np.pi*np.exp(1)*var)

    # mu, sig = model.predict()
    return entropy


def isvalid(node, shape):
    if 0 <= node[0] < shape[0] and 0 <= node[1] < shape[1]:
        return True
    return False


class Node(object):
    def __init__(self, map_pose=None, gval=None, utility=None, parents_index=[],
                 path=np.empty((0, 2))):
        self.map_pose = map_pose
        self.gval = gval
        self.camera_utility = utility
        self.parents_index = parents_index[:]
        self.sensor_utility = None
        self.path = np.copy(path)


class FieldEnv(object):
    def __init__(self, num_rows=40, num_cols=40):
        super(FieldEnv, self).__init__()

        self.num_rows, self.num_cols = num_rows, num_cols
        self.shape = (self.num_rows, self.num_cols)
        self.X, self.Y = generate_gaussian_data(num_rows, num_cols)

        # Map of Field
        num_row_pass = 4
        assert num_rows % (num_row_pass + 1) == 0, 'Infeasible row setting'
        self.map = Map(num_rows, num_cols, num_row_pass, row_pass_width=1)

    def collect_samples(self, indices, noise_std):
        y = self.Y[indices] + np.random.normal(0, noise_std, size=len(indices))
        return y

    # def _get_state(self):
    #     mu, std = self.gp.predict(self.x, return_std=True)
    #     pose = tuple(self.train_x[-1,:])
    #     mae = np.mean(np.abs(mu - self.y))
    #
    #     self.state = {
    #     'variance': np.reshape(std**2, (self.num_rows, self.num_cols)),
    #     'y_pred' : np.reshape(mu, (self.num_rows, self.num_cols)),
    #     'error' : mae,
    #     'pose' : pose
    #     }
    #     return self.state
    #
    # def _update(self, x, y):
    #     x = np.reshape(x,(-1,self.train_x.shape[1]))
    #     y = np.reshape(y,-1)
    #     self.train_x = np.concatenate([self.train_x, x])
    #     self.train_y = np.concatenate([self.train_y, y])
    #     self.train()
    #
    #
    # def step(self, action):
    #     # action - should be a numpy array in [0,1)
    #     x = np.array([action[0]*self.num_rows, action[1]*self.num_cols]).astype(int)
    #     x = np.maximum([0,0], np.minimum(x, [self.num_rows-1, self.num_cols-1]))
    #     y = self.env.sample(x)
    #
    #     previous_state = deepcopy(self.state)
    #     self._update(x, y)
    #     state = self._get_state()
    #     reward = previous_state['error'] - state['error']
    #     return state, reward
    #
    # def expert_action(self, state=None):
    #     if state is None:
    #         state = self.state
    #     var = state['variance']
    #     pose = np.array(state['pose'])
    #     # all locations with maximum variance
    #     tx = self.x[np.where(var.flatten()==var.max())[0],:]
    #     dist = np.sum(np.abs(tx - pose.reshape(1,-1)),axis=1)
    #     pos = tx[np.argmax(dist)]
    #     return np.array([pos[0]/self.num_rows, pos[1]/self.num_rows])
    #
    #
    # def draw_plots(self):
    #     plt.figure(0)
    #     plt.title('True Values')
    #     plt.imshow(self.y.reshape(self.num_rows, self.num_cols))
    #
    #     plt.figure(1)
    #     plt.title('Predicted Values')
    #     plt.imshow(self.state['y_pred'])
    #
    #     plt.figure(2)
    #     plt.title('Variance')
    #     plt.imshow(self.state['variance'])


if __name__ == '__main__':
    env = FieldEnv(num_rows=10, num_cols=10)
    agent = Agent(env, sensor_model_type='GP', camera_model_type='GP')
    agent.run(render=True)
