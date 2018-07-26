import ipdb
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import mi_change, entropy_from_cov, conditional_entropy, is_valid_cell
from env import FieldEnv
from models import SklearnGPR, GpytorchGPR


class Agent(object):
    def __init__(self, env, model_type='gpytorch_GP', **kwargs):
        super()
        self.env = env
        self.gp_type = model_type
        self.gp = None
        self._init_models(**kwargs)

        self._camera_noise = 1.0
        self._sensor_noise = 0.05
        utility_types = ['entropy', 'information_gain']
        self.utility_type = utility_types[1]
        all_strategies = ['sensor_maximum_utility',
                          'camera_maximum_utility',
                          'informative']
        self._strategy = all_strategies[2]
        var_methods = ['sum', 'max']
        self._var_method = var_methods[1]

        self._visited = np.full(env.num_samples, False)
        self.obs_y = np.zeros(env.num_samples)
        self.obs_var_inv = np.zeros(env.num_samples)

        self._pre_train(num_samples=20, only_sensor=True)
        self.agent_map_pose = (0, 0)
        self.search_radius = 10
        # self.mi_radius = self.search_radius*3
        self.mi_radius = np.inf

        self.path = np.copy(self.agent_map_pose).reshape(-1, 2)
        self.sensor_seq = np.empty((0, 2))
        self.gp_update_every = 0
        self.last_update = 0

    def _init_models(self, **kwargs):
        if self.gp_type == 'sklearn_GP':
            self.gp = SklearnGPR()
        elif self.gp_type == 'gpytorch_GP':
            latent = kwargs['latent'] if 'latent' in kwargs else None
            lr = kwargs['lr'] if 'lr' in kwargs else .01
            self.gp = GpytorchGPR(latent, lr)
        else:
            raise NotImplementedError

    def _pre_train(self, num_samples, only_sensor=False):
        ind = np.random.randint(0, self.env.num_samples, num_samples)
        if only_sensor:
            self.add_samples(ind, self._sensor_noise)
        else:
            self.add_samples(ind[:num_samples//2], self._camera_noise)
            self.add_samples(ind[num_samples//2:], self._sensor_noise)
        self.update_model()
        
    def add_samples(self, indices, noise):
        y = self.env.collect_samples(indices, noise)
        y_noise = np.full(len(indices), noise)
        self._handle_data(indices, y, y_noise)
        self._visited[indices] = True

    def _compute_new_var(self, var_inv, old_var_inv):
        if self._var_method == 'sum':
            return var_inv + old_var_inv
        elif self._var_method == 'max':
            return np.maximum(var_inv, old_var_inv)
        else:
            raise NotImplementedError

    def _handle_data(self, indices, y, var):
        var_inv = 1.0/np.array(var)
        old_obs = self.obs_y[indices]
        old_var_inv = self.obs_var_inv[indices]
        new_obs = (old_obs * old_var_inv + y * var_inv)/(old_var_inv + var_inv)
        new_var_inv = self._compute_new_var(var_inv, old_var_inv)
        self.obs_y[indices] = new_obs
        self.obs_var_inv[indices] = new_var_inv

    def update_model(self):
        x = self.env.X[self._visited]
        var = 1.0/(self.obs_var_inv[self._visited])
        y = self.obs_y[self._visited]
        self.gp.fit(x, y, var)

    def run(self, render=False, iterations=40):
        if self._strategy == 'sensor_maximum_utility':
            raise NotImplementedError
            # self.step_max_utility('sensor', render, iterations)
        elif self._strategy == 'camera_maximum_utility':
            raise NotImplementedError
            # self.step_max_utility('camera', render, iterations)
        elif self._strategy == 'informative':
            self.run_informative(render, iterations)
        else:
            raise NotImplementedError

    def _render_path(self, ax):
        # ax.set_cmap('hot')
        plot = 1.0 - np.repeat(self.env.map.oc_grid[:, :, np.newaxis], 3, axis=2)
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
        # render path
        self._render_path(ax[0, 0])

        # ipdb.set_trace()

        # render plots
        axt, axp, axv = ax[1, 0], ax[1, 1], ax[0, 1]
        axt.set_title('Ground Truth')
        imt = axt.imshow(self.env.Y.reshape(self.env.shape))
        div = make_axes_locatable(axt)
        caxt = div.new_horizontal(size='5%', pad=.05)
        # fig.add_axes(caxt)
        # fig.colorbar(imt, caxt, orientation='vertical')

        axp.set_title('Predicted values')
        imp = axp.imshow(pred.reshape(self.env.shape))
        divm = make_axes_locatable(axp)
        caxp = divm.new_horizontal(size='5%', pad=.05)
        # fig.add_axes(caxp)
        # fig.colorbar(imp, caxp, orientation='vertical')

        axv.set_title('Variance')
        imv = axv.imshow(var.reshape(self.env.shape))

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
    #         mask = np.copy(self.sensor__visited.flatten())
    #     elif source == 'camera':
    #         model = self.camera_model
    #         mask = np.copy(self.camera__visited.flatten())
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
    #         info = mi_change(x, model, A, A_bar, model.train_var)
    #         mi[i] = info
    #
    #     gp_indices = np.where(mi == mi.max())[0]
    #     map_poses = self.env.gp_index_to_map_pose(gp_indices)
    #     distances = self.env.map.get_distances(self.map_pose, map_poses)
    #     idx = np.argmin(distances)
    #     return gp_indices[idx], map_poses[idx]

    def predict(self):
        pred, std = self.gp.predict(self.env.X, return_std=True)
        return pred, std**2

    def run_informative(self, render, iterations):
        if render:
            plt.ion()
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        for i in range(iterations):
            # find next node to visit
            next_node = self._bfs_search(self.agent_map_pose, self.search_radius)
            
            # add samples (camera and sensor)
            self.add_samples(next_node.parents_index, self._camera_noise)
            gp_index = self.env.map_pose_to_gp_index(next_node.map_pose)
            if gp_index is not None:
                self.add_samples([gp_index], self._sensor_noise)

            # update GP model
            self.update_model()

            # update agent history
            self.path = np.concatenate([self.path, next_node.path], axis=0).astype(int)
            self.agent_map_pose = next_node.map_pose
            if gp_index is not None:
                self.sensor_seq = np.concatenate(
                    [self.sensor_seq, np.array(self.agent_map_pose).reshape(-1, 2)]).astype(int)

            pred, var = self.predict()
            # if np.max(var) < .01:
            #     print('Converged')
            #     break

            if render:
                # fig.clf()
                self.render(fig, ax, pred, var)
                plt.pause(.1)
                # plt.show()
        ipdb.set_trace()

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

                    # directional BFS (can't move in the opposite direction)
                    # do not expand nodes in the opposite direction
                    if len(self.path) > 2 and node.map_pose == self.agent_map_pose and new_map_pose == tuple(self.path[-2]):
                            continue

                    if gp_index is None:
                        new_parents_index = node.parents_index
                    else:
                        new_parents_index = node.parents_index + [gp_index]

                    new_path = np.concatenate([node.path, np.array(new_map_pose).reshape(-1, 2)])
                    new_node = Node(new_map_pose, new_gval, node.utility,
                                    new_parents_index, new_path)
                    open_nodes.append(new_node)

        # NOTE: computational bottleneck
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
            # node.utility = self._get_utility(indices, x_noise)
            all_nodes_indices.append(indices)
            all_nodes_x_noise.append(x_noise)
        # total_utility = [node.utility for node in closed_nodes]
        # best_node = closed_nodes[np.argmax(total_utility).item()]

        # computing mutual information
        if self.mi_radius > np.prod(self.env.map.shape):
            v_ind = list(range(self.env.num_samples))
        else:
            v_ind = self.env.get_neighborhood(self.agent_map_pose, self.mi_radius)[1]
        # these indices have to be excluded for mi computation
        rest = list(set(np.arange(self.env.num_samples)) - set(v_ind))

        # NOTE: this only needs to be computed after updating model once
        v = self.env.X[v_ind, :]
        # cov = self.gp.cov_mat(self.env.X, self.env.X, white_noise=None)
        cov_v = self.gp.cov_mat(v, v, white_noise=None)
        info = []

        for indices, noise in zip(all_nodes_indices, all_nodes_x_noise):
            var_inv = np.copy(self.obs_var_inv)
            var_inv[indices] = self._compute_new_var(var_inv[indices], 1.0 / np.array(noise))
            var_inv_v = var_inv[v_ind]

            vis = np.copy(self._visited)
            vis[indices] = True
            vis[rest] = False

            # cov_a = cov[vis].T[vis].T
            # a_noise = 1.0/var_inv[vis]
            cov_a = cov_v[vis[v_ind]].T[vis[v_ind]].T
            a_noise = 1.0/var_inv_v[vis[v_ind]]

            # entropy of new A
            e1 = entropy_from_cov(cov_a + np.diag(a_noise))

            vis[rest] = True
            unvis = ~vis
            # var_inv[unvis] = self._camera_noise
            # cov_a_bar = cov[unvis].T[unvis].T
            # a_bar_noise = 1.0/var_inv[unvis]
            var_inv_v[unvis[v_ind]] = self._camera_noise
            cov_a_bar = cov_v[unvis[v_ind]].T[unvis[v_ind]].T
            a_bar_noise = 1.0/var_inv_v[unvis[v_ind]]

            # NOTE: without noise term, the determinants of both the covariance
            # matrices reduce to 0, also sensitive to noise
            # entropy of new A_bar (V - A)
            e2 = entropy_from_cov(cov_a_bar + np.diag(a_bar_noise))
            # entropy of V
            # cov_v = cov[v_ind, :][:, v_ind]
            # v_noise = 1.0/var_inv[v_ind]
            v_noise = 1.0/var_inv_v
            e3 = entropy_from_cov(cov_v + np.diag(v_noise))

            info.append(e1 + e2 - e3)

        best_node = closed_nodes[np.argmax(info).item()]
        return best_node

    # @deprecated (implemented a faster way instead)
    # def _get_utility(self, indices, x_noise_var):
    #     x = self.env.X[indices, :]
    #     a = self.gp.train_x
    #     a_noise_var = self.gp.train_var
    #     if self.utility_type == 'information_gain':
    #         a_bar = self.env.X[~self._visited]
    #         info = mi_change(x, a, a_bar, self.gp,
    #                          x_noise_var, a_noise_var,
    #                          a_bar_noise_var=None)
    #     elif self.utility_type == 'entropy':
    #         info = conditional_entropy(x, a, self.gp,
    #                                    x_noise_var, a_noise_var)
    #     else:
    #         raise NotImplementedError
    #     return info


class Node(object):
    def __init__(self, map_pose, gval, utility, parents_index, path=np.empty((0, 2))):
        self.map_pose = map_pose
        self.gval = gval
        self.utility = utility
        self.parents_index = parents_index[:]
        self.path = np.copy(path)


if __name__ == '__main__':
    data_file = 'data/plant_width_mean_dataset.pkl'
    env = FieldEnv(data_file=data_file)
    agent = Agent(env, model_type='gpytorch_GP')
    # agent = Agent(env, model_type='sklearn_GP')
    agent.run(render=True)