import numpy as np
import matplotlib.pyplot as plt 
import ipdb
from utils import manhattan_distance
from graph_utils import get_heading, opposite_headings
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


class Map(object):
    def __init__(self, num_gp_rows=15, num_gp_cols=37, num_row_passes=2, row_pass_width=1):

        super(Map, self).__init__()
        
        self.num_gp_rows = num_gp_rows
        self.num_gp_cols = num_gp_cols
        self.num_row_passes = num_row_passes
        self.row_pass_width = row_pass_width

        assert self.num_gp_rows % (self.num_row_passes + 1) == 0, 'Infeasible row setting'

        self._shape = self._compute_map_dimensions()
        self.stack_len = self.num_gp_rows // (self.num_row_passes + 1)
        self.row_pass_indices = self._get_row_pass_indices()
        self.free_cols = np.arange(0, self.shape[1], 2)
        self.obstacle_cols = np.arange(1, self.shape[1], 2)

        # 1 if obstacle 0 otherwise
        self.occupied = self._set_occupancy_grid()

        # all poses (x,y) in the map
        # x, y = np.meshgrid(np.arange(self._shape[1]), np.arange(self._shape[0]))
        # self.all_poses = np.vstack([y.flatten(), x.flatten()]).T

    @property
    def shape(self):
        return self._shape

    def _set_occupancy_grid(self):
        # returns the occupancy grid of the map
        grid = np.full(self._shape, False)
        nr, nc = self._shape
        grid[:, self.obstacle_cols] = True
        grid[self.row_pass_indices, :] = False
        return grid

    def _compute_map_dimensions(self):
        # extra row at top and bottom
        total_rows = self.num_gp_rows + (self.num_row_passes + 2) * self.row_pass_width
        total_cols = self.num_gp_cols * 2 - 1    
        return total_rows, total_cols

    def _get_row_pass_indices(self):
        # return indices of all the row pass
        ind = []
        t = 0
        last = 0
        while t < self.num_row_passes + 2:
            ind += list(range(last, last + self.row_pass_width))
            t += 1
            last = last + self.row_pass_width + self.stack_len
        return np.array(ind)
   
    def distance_between_nodes(self, start, goal, heading):
        # these cases should never occur
        if start[0] not in self.row_pass_indices:
            if heading not in [(1,0),(-1,0)]:
                ipdb.set_trace()

        if goal[0] in self.row_pass_indices:
            ipdb.set_trace()

        # return distance between start and goal and final heading on goal
        # if start and goal are in the same column
        if start[1] == goal[1]:
            h = get_heading(goal, start)
            opposite = opposite_headings(heading, h)
            # if headings align, then just move to the goal
            if not opposite:
                return manhattan_distance(start, goal), get_heading(goal, start)
            # if not, move to the junction, then move to the adjacent column (and come back later) and proceed to the goal
            else:
                sj = self.get_junction(start, heading)    
                gj = self.get_junction(goal, heading)

                # start and goal in different blocks
                if sj!=gj:
                    total_dist = manhattan_distance(start, sj) + 2*2 + manhattan_distance(sj, goal)
                    return total_dist, (-heading[0], 0)
                # start and goal in same block, move to the opposite junction first and then straight to goal
                else:
                    node = self.get_junction(goal, (-heading[0], 0))
                    total_dist = manhattan_distance(start, sj) + 2*2 + manhattan_distance(sj, node) + manhattan_distance(node, goal)
                    return total_dist, heading

        # start and goal are in different columns
        # move to the junction and then proceed to the goal
        else:
            if heading in [(1,0), (-1,0)]:
                node = self.get_junction(start, heading)
                total_dist = manhattan_distance(start, node) + manhattan_distance(node, goal)
                # shift to the column which has the goal and compute new heading
                final_heading = get_heading(goal, (node[0], goal[1]))
                final_heading = heading if final_heading is None else final_heading
                return total_dist, final_heading
            else:
                up_node = self.get_up_junction(goal)
                down_node = self.get_down_junction(goal)
                # go to down node if up node lies in the same row as start
                if start[1] == up_node[1]:
                    total_dist = manhattan_distance(start, down_node) + manhattan_distance(down_node, goal)
                    final_heading = (-1,0)
                # go to up node if down node lies in the same row as start
                elif start[1] == down_node[1]:
                    total_dist = manhattan_distance(start, up_node) + manhattan_distance(up_node, goal)
                    final_heading = (1,0)
                else:
                    total_dist = manhattan_distance(start, goal)
                    final_heading = get_heading(goal, (start[0], goal[1]))
                return total_dist, final_heading

    def get_junction(self, pose, heading):
        # return junction in the heading direction
        if heading == (1,0):
            return self.get_down_junction(pose)
        elif heading == (-1,0):
            return self.get_up_junction(pose)
        else:
            return pose
            # print(pose, heading)
            raise NotImplementedError

    def get_up_junction(self, pose):
        # return up junction (in decreasing x direction)
        up = max([x for x in self.row_pass_indices if x<=pose[0]])
        return (up, pose[1])

    def get_down_junction(self, pose):
        # return down junction (in increasing x direction)
        down = min([x for x in self.row_pass_indices if x>=pose[0]])
        return (down, pose[1])

    def nearest_waypoint_path_cost(self, start, start_heading, waypoints):
        # return cost of the path formed by always moving to the nearest waypoint
        nw = len(waypoints)
        visited = [False]*nw
        total_cost = 0

        node = start
        heading = start_heading
        while sum(visited) != nw:
            # find the nearest waypoint from the current node
            all_dists = [np.inf]*nw
            all_final_headings = [(0,0)]*nw
            for i in range(nw):
                if visited[i]:
                    continue
                dist, final_heading = self.distance_between_nodes(node, waypoints[i], heading)
                all_dists[i] = dist
                all_final_headings[i] = final_heading

            idx = np.argmin(all_dists)
            total_cost += all_dists[idx]
            node = waypoints[idx]
            heading = all_final_headings[idx]
            visited[idx] = True
        return total_cost

    # def _render_path(self, ax):
    #     # ax.set_cmap('hot')
    #     plot = 1.0 - np.repeat(self.env.map.occupied[:, :, np.newaxis], 3, axis=2)
    #     # highlight camera measurement
    #     if self.path.shape[0] > 0:
    #         plot[self.path[:, 0], self.path[:, 1], :] = [.75, .75, .5]
    #     # highlight sensor measurement
    #     if self.sensor_seq.shape[0] > 0:
    #         plot[self.sensor_seq[:, 0], self.sensor_seq[:, 1]] = [.05, 1, .05]

    #     plot[self.agent_map_pose[0], self.agent_map_pose[1], :] = [0, 0, 1]
    #     ax.set_title('Environment')
    #     ax.imshow(plot)

    # def render(self, fig, ax, pred, var):
    #     # self._render_path(ax[0, 0])

    #     # render plots
    #     axt, axp, axv = ax[1, 0], ax[1, 1], ax[0, 1]
    #     axt.set_title('Ground Truth / Actual values')
    #     imt = axt.imshow(self.env.full_Y, cmap='ocean', vmin=0, vmax=self.env.Y.max())
    #     div = make_axes_locatable(axt)
    #     caxt = div.new_horizontal(size='5%', pad=.05)
    #     fig.add_axes(caxt)
    #     fig.colorbar(imt, caxt, orientation='vertical')

    #     axp.set_title('Predicted values')
    #     res_pred = self.env.vec_to_gp_mat(pred)
    #     imp = axp.imshow(res_pred, cmap='ocean', vmin=0, vmax=self.env.Y.max())
    #     divm = make_axes_locatable(axp)
    #     caxp = divm.new_horizontal(size='5%', pad=.05)
    #     fig.add_axes(caxp)
    #     fig.colorbar(imp, caxp, orientation='vertical')

    #     axv.set_title('Variance')
    #     res_var = self.env.vec_to_gp_mat(var)
    #     imv = axv.imshow(res_var, cmap='hot')
    #     # divv = make_axes_locatable(axv)
    #     # caxv = divv.new_horizontal(size='5%', pad=.05)
    #     # fig.add_axes(caxv)
    #     # fig.colorbar(imv, caxv, orientation='vertical')


if __name__ == '__main__':
    small_map = Map(num_gp_rows=15, num_gp_cols=15, num_row_passes=2, row_pass_width=1)

    waypoints = [(7,6), (15,16), (11,10), (5,10)]
    for w in waypoints:
        print(w, small_map.map_pose_to_gp_pose(w), small_map.occupied[w])
    
    start = (0,0)
    heading = (1,0)
    path_length = small_map.nearest_waypoint_path_cost(start, heading, waypoints)


