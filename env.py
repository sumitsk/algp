import numpy as np
import ipdb

from map import Map
from utils import load_data, zero_mean_unit_variance


class FieldEnv(object):
    def __init__(self, num_rows=15, num_cols=15, data_file=None):
        super(FieldEnv, self).__init__()

        if data_file is None:
            from utils import generate_gaussian_data, generate_mixed_data
            self.num_rows, self.num_cols = num_rows, num_cols
            self.X, self.Y = generate_gaussian_data(num_rows, num_cols)
            # self.X, self.Y = generate_mixed_data(num_rows, num_cols)

        else:
            self.num_rows, self.num_cols, self.X, self.Y = load_data(data_file)
            
            # using only a subset of data
            self.num_cols = 12
            self.X = self.X[:self.num_rows*self.num_cols, :]
            self.Y = self.Y[:self.num_rows*self.num_cols]

        self.map = Map(self.num_rows, self.num_cols)
        # TODO: inaccurate 
        self.locs = self.X[:,:2]
        # occ = np.copy(self.map.occupied)
        # occ[self.map.row_pass_indices] = True
        # ipdb.set_trace()
        self._normalize_dataset()
        # Occupancy map of the field
        
    def _normalize_dataset(self):
        # NOTE: this is fine since we know in advance about the samples
        self.X = self.X.astype(float)
        self.X[:, :2] = zero_mean_unit_variance(self.X[:,:2])
        
    def collect_samples(self, indices, noise_std):
        y = self.Y[indices] + np.random.normal(0, noise_std, size=len(indices))
        return y

    @property
    def shape(self):
        return self.num_rows, self.num_cols

    @property
    def num_samples(self):
        return self.X.shape[0]

    def gp_index_to_map_pose(self, gp_index):
        gp_pose = self.gp_index_to_gp_pose(gp_index)
        map_pose = self.map.gp_pose_to_map_pose(gp_pose)
        return map_pose

    def map_pose_to_gp_index(self, map_pose):
        assert isinstance(map_pose, tuple), 'Map pose must be a tuple'
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

    def get_neighborhood(self, map_pose, radius):
        pose = np.array(map_pose).reshape(-1, 2)
        # manhattan distance
        dists = np.sum(np.abs(self.map.all_poses - pose), axis=1)
        mask = (dists <= radius)
        valid_map_poses = self.map.all_poses[np.where(mask)[0]]
        valid_gp_indices = []
        # convert to gp index
        for map_pose in valid_map_poses:
            gp_index = self.map_pose_to_gp_index(tuple(map_pose))
            if gp_index is not None:
                valid_gp_indices.append(gp_index)
        return valid_map_poses, valid_gp_indices

