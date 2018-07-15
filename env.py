import numpy as np
import ipdb

from map import Map
from utils import generate_gaussian_data


class FieldEnv(object):
    def __init__(self, num_rows=40, num_cols=40):
        super(FieldEnv, self).__init__()

        self.num_rows, self.num_cols = num_rows, num_cols
        self.X, self.Y = generate_gaussian_data(num_rows, num_cols)

        # Map of Field
        num_row_pass = 4
        assert num_rows % (num_row_pass + 1) == 0, 'Infeasible row setting'
        self.map = Map(num_rows, num_cols, num_row_pass, row_pass_width=1)

    def collect_samples(self, indices, noise_std):
        # ipdb.set_trace()
        y = self.Y[indices] + np.random.normal(0, noise_std, size=len(indices))
        return y

    @property
    def shape(self):
        return self.num_cols, self.num_cols

    @property
    def num_samples(self):
        return self.X.shape[0]

    def gp_index_to_map_pose(self, gp_index):
        gp_pose = self.gp_index_to_gp_pose(gp_index)
        map_pose = self.map.gp_pose_to_map_pose(gp_pose)
        return map_pose

    def map_pose_to_gp_index(self, map_pose):
        assert isinstance(map_pose, tuple), 'Map pose should be a tuple'
        gp_pose = self.map.map_pose_to_gp_pose(map_pose)
        if gp_pose is None:
            return None
        return self.gp_pose_to_gp_index(gp_pose)

    def gp_pose_to_gp_index(self, gp_pose):
        if isinstance(gp_pose, tuple):
            tmp = np.array(gp_pose)
        else:
            tmp = np.copy(gp_pose)
        indices = np.ravel_multi_index(tmp.reshape(-1, 2).T, self.shape)
        if isinstance(gp_pose, tuple):
            return indices[0]
        return indices

    def gp_index_to_gp_pose(self, gp_index):
        return np.vstack(np.unravel_index(gp_index, self.shape)).T

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



