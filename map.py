import numpy as np
import matplotlib.pyplot as plt 
import ipdb


# defines the occupancy map of the layout
class Map(object):
    def __init__(self,
                 num_gp_rows=15,
                 num_gp_cols=37,
                 num_row_passes=2,
                 row_pass_width=1
                 ):

        super(Map, self).__init__()
        
        self.num_gp_rows = num_gp_rows
        self.num_gp_cols = num_gp_cols
        self.num_row_passes = num_row_passes
        self.row_pass_width = row_pass_width

        assert self.num_gp_rows % (self.num_row_passes + 1) == 0, 'Infeasible row setting'

        self._shape = self._compute_map_dimensions()
        self.stack_len = self.num_gp_rows // (self.num_row_passes + 1)
        self.row_pass_indices = self._get_row_pass_indices()
        
        # 1 if obstacle 0 otherwise
        self.oc_grid = self._build_oc_grid()

        # all poses (x,y) in the map
        x, y = np.meshgrid(np.arange(self._shape[1]), np.arange(self._shape[0]))
        self.all_poses = np.vstack([y.flatten(), x.flatten()]).T

    @property
    def shape(self):
        return self._shape

    def _build_oc_grid(self):
        # generates the occupancy grid
        grid = np.zeros(self._shape)
        nr, nc = self._shape
        grid[:, 1:nc:2] = 1
        grid[self.row_pass_indices, :] = 0
        return grid

    def _compute_map_dimensions(self):
        # extra row at top and bottom
        total_rows = self.num_gp_rows + (self.num_row_passes + 2) * self.row_pass_width
        total_cols = self.num_gp_cols * 2 - 1    
        return total_rows, total_cols

    def _get_row_pass_indices(self):
        ind = []
        t = 0
        last = 0
        while t < self.num_row_passes + 2:
            ind += list(range(last, last + self.row_pass_width))
            t += 1
            last = last + self.row_pass_width + self.stack_len
        return np.array(ind)
        # this was nice but deprecated now
        # c = 0
        # while t < self.num_gp_rows:
        #     if t != 0 and t % self.stack_len == 0:
        #         for j in range(self.row_pass_width):
        #             ind.append(t+j+c)
        #         c += self.row_pass_width
        #
        #     t += 1
        # return np.array(ind[:self.row_pass_width * self.num_row_passes])

    def render(self, gp_pose=None, fig_idx=None):
        fig_idx = 0 if fig_idx is None else fig_idx
        plt.figure(fig_idx)
        plt.title('Field Map')
        plt.imshow(1-self.oc_grid, cmap='gray', interpolation='none', aspect='equal')
        if gp_pose is not None:
            map_pos = self.gp_pose_to_map_pose(gp_pose)
            plt.scatter(map_pos[1], map_pos[0])

        plt.show()

    def gp_pose_to_map_pose(self, gp_pose):
        if isinstance(gp_pose, tuple):
            poses = np.array(gp_pose)
        else:
            poses = np.copy(gp_pose)
        poses = poses.reshape(-1, 2)
        map_poses = np.zeros(poses.shape)
        map_poses[:, 1] = 2 * poses[:, 1]
        a = poses[:, 0] // self.stack_len
        b = poses[:, 0] % self.stack_len + self.row_pass_width
        map_poses[:, 0] = a * (self.stack_len + self.row_pass_width) + b
        map_poses = map_poses.astype(int)
        if isinstance(gp_pose, tuple):
            return tuple(map_poses.squeeze())
        else:
            return map_poses

    def map_pose_to_gp_pose(self, map_pose):
        assert isinstance(map_pose, tuple), 'Map pose must be a tuple!'
        if map_pose[1] % 2 == 1:
            return None
        if map_pose[0] in self.row_pass_indices:
            return None

        y = map_pose[1] // 2
        x = map_pose[0] - np.sum(self.row_pass_indices < map_pose[0])
        return x, y

    @staticmethod
    def get_distances(pose, map_poses):
        pose_ = np.array(pose).reshape(-1, 2)
        map_poses_ = np.array(map_poses).reshape(-1, 2)
        return np.abs(pose_ - map_poses_).sum(axis=-1)

    # @deprecated
    # def get_all_gp_distances(self, pos):
    #     ocgrid_pos = self.gp_pose_to_map_pose(pos)
    #     gvals = run_bfs(self.oc_grid, ocgrid_pos)
    #     temp = gvals[:,0:gvals.shape[1]:2]
    #     temp = np.delete(temp, self.row_pass_indices, axis=0)
    #     return temp


# @deprecated
# class OpenList(object):
#     def __init__(self):
#         super(OpenList, self).__init__()
#         self.nodes = None
#         self.gvals = None
#
#
# def run_bfs(grid, pos):
#     sz = grid.shape
#     dxdy = [(-1,0),(1,0),(0,1),(0,-1)]
#
#     # print(pos, sz)
#     node = np.ravel_multi_index(pos, sz)
#     # openlist = {'nodes':[node], 'gvals':[0]}
#     openlist = OpenList()
#     openlist.nodes = [node]
#     openlist.gvals = [0]
#
#     cost = 1
#     gvals = np.ones(sz) * float('inf')
#     gvals[pos] = 0
#
#     while len(openlist.nodes)!=0:
#         node = openlist.nodes.pop(0)
#         node = np.unravel_index(node, sz)
#         gval = openlist.gvals.pop(0)
#
#         for dx, dy in dxdy:
#             new_node = (node[0]+dx,node[1]+dy)
#             new_gval = gval + cost
#
#             if isvalid(new_node, sz) and grid[new_node] != 1 and new_gval < gvals[new_node]:
#                 gvals[new_node] = new_gval
#                 openlist.nodes.append(np.ravel_multi_index(new_node, sz))
#                 openlist.gvals.append(new_gval)
#     return gvals


def isvalid(node, shape):
    if 0 <= node[0] < shape[0] and 0 <= node[1] < shape[1]:
        return True
    return False


if __name__ == '__main__':
    small_map = Map(num_gp_rows=4,
                    num_gp_cols=4,
                    num_row_passes=1,
                    row_pass_width=1)

    large_map = Map(num_gp_rows=84,
                    num_gp_cols=84,
                    num_row_passes=5,
                    row_pass_width=2)

    small_map.render()
    # for pose in small_map.all_poses:
    #     gp_pose = small_map.map_pose_to_gp_pose(tuple(pose))
    #     print(pose, gp_pose)

    


