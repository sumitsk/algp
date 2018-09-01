import numpy as np
import matplotlib.pyplot as plt 
import ipdb
from utils import manhattan_distance, BFSNode, is_valid_cell
from copy import deepcopy
from graph_utils import get_heading, opposite_headings


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
        self.obstacle_cols = np.arange(0, self.shape[1], 2)
        
        # 1 if obstacle 0 otherwise
        self.occupied = self._build_occupancy_grid()

        # all poses (x,y) in the map
        x, y = np.meshgrid(np.arange(self._shape[1]), np.arange(self._shape[0]))
        self.all_poses = np.vstack([y.flatten(), x.flatten()]).T

    @property
    def shape(self):
        return self._shape

    def _build_occupancy_grid(self):
        # generates the occupancy grid
        grid = np.full(self._shape, False)
        nr, nc = self._shape
        grid[:, 1:nc:2] = True
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

    # TODO: make this the defacto rendering method and remove others
    def render(self, gp_pose=None, fig_idx=None, paths=None, locs=None):
        fig_idx = 0 if fig_idx is None else fig_idx
        plt.figure(fig_idx)
        plt.title('Field Map')
        plot = 1.0 - np.repeat(self.occupied[:, :, np.newaxis], 3, axis=2)
        if gp_pose is not None:
            map_pos = self.gp_pose_to_map_pose(gp_pose)
            plt.scatter(map_pos[1], map_pos[0])

        if paths is not None:
            plot = 1.0 - np.repeat(self.occupied[:, :, np.newaxis], 3, axis=2)
            # highlight all paths
            for path in paths:
                plot[path[:, 0], path[:, 1], :] = [.75, .75, .5]

        plt.imshow(plot, cmap='gray', interpolation='none', aspect='equal')
        plt.show()

    def gp_pose_to_map_pose(self, gp_pose):
        # convert gp_pose to map_pose 
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
        # convert map pose to gp pose 
        assert isinstance(map_pose, tuple), 'Map pose must be a tuple!'
        if map_pose[1] % 2 == 1:
            return None
        if map_pose[0] in self.row_pass_indices:
            return None

        y = map_pose[1] // 2
        x = map_pose[0] - np.sum(self.row_pass_indices < map_pose[0])
        return x, y

    # TODO: remove if not needed anymore
    @staticmethod
    def get_distances(pose, map_poses):
        raise NotImplementedError('This is wrong')
        # distance between two nodes won't be simply manhattan distance
        # pose_ = np.array(pose).reshape(-1, 2)
        # map_poses_ = np.array(map_poses).reshape(-1, 2)
        # return np.abs(pose_ - map_poses_).sum(axis=-1)

    def distance_between_nodes(self, start, goal, heading):
        # return distance between start and goal and final heading on goal

        # if start and goal are in the same column
        if start[1] == goal[1]:
            h = get_heading(goal, start)
            opposite = opposite_headings(heading, h)
            # if headings align, then just move to the goal
            if not opposite:
                return manhattan_distance(start, goal), heading
            # if not, move to the junction, then move to the adjacent column (and come back later) and proceed to the goal
            else:
                node = self.get_junction(start, heading)    
                total_dist = manhattan_distance(start, node) + 2*2 + manhattan_distance(node, goal)
                return total_dist, (-heading[0], 0)

        # start and goal are in different columns
        # move to the junction and then proceed to the goal
        else:
            node = self.get_junction(start, heading)
            total_dist = manhattan_distance(start, node) + manhattan_distance(node, goal)
            # shift to the column which has the goal and compute new heading
            final_heading = get_heading(goal, (node[0], goal[1]))
            final_heading = heading if final_heading is None else final_heading
            return total_dist, final_heading

    def get_junction(self, pose, heading):
        # return junction in the heading direction
        if heading == (1,0):
            return self.get_down_junction(pose)
        elif heading == (-1,0):
            return self.get_up_junction(pose)
        else:
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
        # NOTE: this serves as an upper bound on the path cost
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

    # TODO: remove if not required
    def all_paths(self, start, waypoints, delta_input=None, max_length=None):
        # waypoints - list of tuples
        start = tuple(start)
        n_waypoints = len(waypoints)
        # location of waypoints
        w_mask = np.full(self.shape, False)
        for w in waypoints:
            w_mask[w] = True

        if max_length is None:
            # max_length = 2.5 * minimum_distance_to_go(start, waypoints)
            max_length = 1 * self.min_dist_to_goal(start, waypoints[0], delta_input) + self.stack_len

        cost = 1
        dx_dy = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        node = BFSNode(pose=start, gval=0, visited=np.full(n_waypoints, False))
        open_nodes = [node]
        closed_nodes = []  
        while len(open_nodes) > 0:
            node = open_nodes.pop(0)
            # if node is a valid sampling location, then we do not need to do delta additions here, we can jump to intersection
            
            if len(node.path)>=2:
                delta = node.pose[0] - node.path[-2][0]
            else:
                delta = delta_input

            gp_pose = self.map_pose_to_gp_pose(node.pose)
            if gp_pose is not None:
                # because of no U-turn there will be only one way extension
                
                # move agent until an intersection
                while gp_pose is not None:
                    node.gval += cost
                    if node.gval > max_length:
                        break 
                    assert delta is not None, 'delta can not be None'
                    node.pose = (node.pose[0] + delta, node.pose[1])                    
                    node.path.append(node.pose)
                    if w_mask[node.pose]:
                        node.visited[waypoints.index(node.pose)] = True
                        # print('waypoint visited')
                        # print(node.__dict__)

                    if node.visited.all():
                        closed_nodes.append(node)
                        break
                    gp_pose = self.map_pose_to_gp_pose(node.pose)
                # insert into open nodes only if it has reached an intersection
                if gp_pose is None:
                    open_nodes.append(node)

            else:
                for dxdy in dx_dy:
                    new_pose = (node.pose[0] + dxdy[0], node.pose[1] + dxdy[1])

                    # within bounds and unoccupied
                    if not is_valid_cell(new_pose, self.shape) or self.occupied[new_pose]:
                        continue

                    # don't expand in the reverse direction (robot can't take a U-turn)
                    if len(node.path)>=2 and new_pose==node.path[-2]:
                        continue

                    new_gval = node.gval + cost 

                    # for only 1 waypoint (essentially if there is a goal location)
                    # to_go = self.min_dist_to_goal(new_pose, waypoints[0], delta) > max_length
                    to_go = manhattan_distance(new_pose, waypoints[0])
                    if new_gval + to_go > max_length:
                        continue
                    # doesn't exceed maximum length
                    # remaining_waypoints = [loc for i,loc in enumerate(waypoints) if not node.visited[i]]
                    # if new_gval + minimum_distance_to_go(new_pose, remaining_waypoints) > max_length:
                    #     continue
                    
                    # if there is a loop and it doesn't contain any waypoint then remove it
                    # NOTE: an intersection can never be a waypoint
                    if new_pose in node.path:
                        # last index of new_pose in node.path
                        idx = len(node.path) - 1 - node.path[::-1].index(new_pose)
                        loop = node.path[idx:]
                        if not has_waypoints(loop, waypoints):
                            continue

                    new_visited = np.copy(node.visited)
                    new_path = deepcopy(node.path)
                    new_path.append(new_pose)
                    new_node = BFSNode(new_pose, new_gval, new_visited, new_path)
                    if w_mask[new_node.pose]:
                        new_node.visited[waypoints.index(new_node.pose)] = True
                        # print('waypoint reached')
                        # print(new_node.__dict__)

                    if new_node.visited.all():
                        closed_nodes.append(new_node)
                    else:
                        open_nodes.append(new_node)

        paths = [np.stack(node.path) for node in closed_nodes]
        # there should not be any duplicates
        # if duplicate_paths(paths):
        #     ipdb.set_trace()
        # ipdb.set_trace()
        return paths

    # not required anymore
    def find_loops(self, path):
        visited = np.full(self.shape, False)
        for node in path:
            # loop detected
            if visited[tuple(node)] is True:
                ipdb.set_trace()
            visited[tuple(node)] = True
        print(len(path), visited.sum())

    # TODO: remove if not required
    def min_dist_to_goal(self, pose, goal, delta=None):
        if delta is None:
            return manhattan_distance(pose, goal)

        if pose[0] in self.row_pass_indices:
            if pose[1]%2==1:
                return manhattan_distance(pose, goal)
            else:
                next_pose = (pose[0] + delta, pose[1])
                return manhattan_distance(next_pose, goal) + 1

        x = pose[0]
        while x not in self.row_pass_indices:
            x += delta
        next_pose = (x, pose[1])
        return abs(x-pose[0]) + manhattan_distance(next_pose, goal)
    
        # find downward intersection
        # if delta==1:

        # # find upward intersection
        # elif delta==-1:

        # else:
        #     raise NotImplementedError


def has_waypoints(path, waypoints):
    if waypoints is None:
        return False
    for loc in waypoints:
        if loc in path:
            return True
    return False


def duplicate_paths(paths):
    # paths is a list of numpy arrays
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            p1 = paths[i]
            p2 = paths[j]
            if p1.shape == p2.shape and (p1==p2).all():
                return True
    return False


if __name__ == '__main__':
    small_map = Map(num_gp_rows=15,
                    num_gp_cols=15,
                    num_row_passes=2,
                    row_pass_width=1)

    waypoints = [(7,6), (15,16), (11,10), (5,10)]
    for w in waypoints:
        print(w, small_map.map_pose_to_gp_pose(w), small_map.occupied[w])
    
    start = (0,0)
    heading = (1,0)
    path_length = small_map.nearest_waypoint_path_cost(start, heading, waypoints)
    ipdb.set_trace()


