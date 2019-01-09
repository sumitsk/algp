import numpy as np
from utils import manhattan_distance
from graph_utils import get_heading, opposite_headings
import ipdb


class Map(object):
    def __init__(self, num_gp_rows=15, num_gp_cols=37, num_row_passes=2, row_pass_width=1):

        super(Map, self).__init__()
        
        self.num_gp_rows = num_gp_rows
        self.num_gp_cols = num_gp_cols
        self.num_row_passes = num_row_passes
        self.row_pass_width = row_pass_width

        assert self.num_gp_rows % (self.num_row_passes + 1) == 0, 'Infeasible row setting'

        self._shape = self._compute_map_dimensions()
        self.corridor_len = self.num_gp_rows // (self.num_row_passes + 1)
        self.row_pass_indices = self._get_row_pass_indices()
        self.free_cols = np.arange(0, self.shape[1], 2)
        self.obstacle_cols = np.arange(1, self.shape[1], 2)

        # 1 if obstacle 0 otherwise
        self.occupied = self._set_occupancy_grid()

    @property
    def shape(self):
        return self._shape

    def _set_occupancy_grid(self):
        # returns the occupancy grid of the map
        grid = np.full(self._shape, False)
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
            last = last + self.row_pass_width + self.corridor_len
        return np.array(ind)
   
    def distance_between_nodes(self, start, goal, heading):
        # return distance between start and goal and final heading on reaching goal
        if start == goal:
            return 0, heading

        # these cases should never occur
        if start[0] not in self.row_pass_indices and heading not in [(1,0),(-1,0)]:
            raise NotImplementedError('Starting location has infeasible heading')
        if goal[0] in self.row_pass_indices:
            raise NotImplementedError('Goal location is a junction')

        # if start and goal are in the same column
        if start[1] == goal[1]:
            final_heading = get_heading(start, goal)
            
            # if headings align, move to the goal directly
            if not opposite_headings(heading, final_heading):
                return manhattan_distance(start, goal), final_heading
            
            # if not, move to the junction, then move to the adjacent column (and come back later) and proceed to the goal
            else:
                sj = self.get_junction(start, heading)    
                gj = self.get_junction(goal, heading)

                # start and goal are in different blocks
                if sj!=gj:
                    total_dist = manhattan_distance(start, sj) + 2*2 + manhattan_distance(sj, goal)
                    return total_dist, (-heading[0], 0)
                
                # start and goal are in the same block, need to come back in this block 
                else:
                    node = self.get_junction(goal, (-heading[0], 0))
                    total_dist = manhattan_distance(start, sj) + 2*2 + manhattan_distance(sj, node) + manhattan_distance(node, goal)
                    return total_dist, heading

        # start and goal are in different columns
        else:
            # move to the junction and then proceed to the goal
            if heading in [(1,0), (-1,0)]:
                node = self.get_junction(start, heading)
                total_dist = manhattan_distance(start, node) + manhattan_distance(node, goal)
                # shift to the goal column to compute final heading
                final_heading = get_heading((node[0], goal[1]), goal)
                final_heading = heading if final_heading is None else final_heading
                return total_dist, final_heading

            # start location is a junction and heading is either east or west 
            else:
                # if heading points towards the goal direction, just move there 
                if (goal[1] >= start[1] and heading[1] > 0) or (goal[1] <= start[1] and heading[1] < 0):
                    total_dist = manhattan_distance(start, goal)
                    # shift to the goal column to compute final heading
                    final_heading = get_heading((start[0], goal[1]), goal)
                    return total_dist, final_heading

                # if heading points in the opposite direction of goal
                else:
                    up_node = self.get_up_junction(goal)
                    down_node = self.get_down_junction(goal)
     
                    # go to down node if up node lies in the same row as start
                    if start[0] == up_node[0]:
                        total_dist = manhattan_distance(start, down_node) + manhattan_distance(down_node, goal)
                        final_heading = (-1,0)
                    # go to up node if down node lies in the same row as start
                    elif start[0] == down_node[0]:
                        total_dist = manhattan_distance(start, up_node) + manhattan_distance(up_node, goal)
                        final_heading = (1,0)
                    else:
                        total_dist = manhattan_distance(start, goal)
                        final_heading = get_heading((start[0], goal[1]), goal)
                    return total_dist, final_heading

    def distance_between_nodes_with_headings(self, start, start_heading, goal, goal_heading):
        dist, final_heading = self.distance_between_nodes(start, goal, start_heading)
        if not opposite_headings(final_heading, goal_heading):
            return dist

        # Goal heading is opposite of final_heading
        perimeter = 4 + 2*(self.corridor_len+1)

        if start_heading in [(1,0),(-1,0)]:
            start_junc = self.get_junction(start, start_heading)
            gj = self.get_junction(goal, start_heading)
            goal_junc = self.get_junction(goal, (-goal_heading[0], goal_heading[1]))
            # the agent has to move atleast this distance 
            min_dist = manhattan_distance(start, start_junc) + manhattan_distance(start_junc, goal_junc) + manhattan_distance(goal_junc, goal)

            # if start and goal are in the same column
            if start[1]==goal[1]:
                # same block
                if start_junc[0]==gj[0]:
                    # top or bottom block
                    if start_junc[0]==0 or start_junc[0]==self.shape[0]-1:
                        extra = perimeter + 4
                    else:
                        extra = perimeter
                # different blocks
                else:
                    extra = 4

            # start and goal are in adjacent sampling columns        
            elif abs(start[1]-goal[1])==2:
                if start_heading == goal_heading:
                    extra = 4
                else:
                    extra = 0

            else:
                extra = 0
        else:
            raise NotImplementedError

        return min_dist + extra

    def get_junction(self, pose, heading):
        # return junction in the heading direction
        if heading == (1,0):
            return self.get_down_junction(pose)
        elif heading == (-1,0):
            return self.get_up_junction(pose)
        else:
            return pose
            
    def get_up_junction(self, pose):
        # return up junction (in decreasing x direction)
        up = max([x for x in self.row_pass_indices if x<=pose[0]])
        return (up, pose[1])

    def get_down_junction(self, pose):
        # return down junction (in increasing x direction)
        down = min([x for x in self.row_pass_indices if x>=pose[0]])
        return (down, pose[1])

    def nearest_waypoint_path_cost(self, start, start_heading, waypoints, return_seq=False):
        # return cost of the path formed by always moving to the nearest waypoint
        nw = len(waypoints)
        visited = [False]*nw
        total_cost = 0

        node = start
        heading = start_heading
        if return_seq:
            seq = []
            costs = []
            # final_headings = []

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
            if return_seq:
                costs.append(all_dists[idx])
                seq.append(idx)
                # final_headings.append(heading)

        if return_seq:
            # return costs, seq, final_headings
            return costs, seq
        return total_cost
