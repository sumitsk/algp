import numpy as np
import ipdb

from map import Map
from utils import load_data, zero_mean_unit_variance, is_valid_cell
from networkx import nx
from copy import deepcopy
from graph_utils import get_new_nodes_and_edges, path_cost, upper_bound, edge_cost, get_heading, distance_to_go, node_action

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
            # self.num_cols = 12
            # self.X = self.X[:self.num_rows*self.num_cols, :]
            # self.Y = self.Y[:self.num_rows*self.num_cols]

        self.map = Map(self.num_rows, self.num_cols)
        self._normalize_dataset()
        
        # (i,j) ^{th} position correponds to the index of gp sample at that map location (i,j)
        self.indices_mask = np.full(self.map.shape, None)
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                self.indices_mask[i,j] = self.map_pose_to_gp_index((i,j))

        self._setup_graph()
        self.extra_cost = 2*(self.map.stack_len + 3)

    def _normalize_dataset(self):
        # NOTE: this is fine since we know in advance about the samples
        self.X = self.X.astype(float)
        self.X[:, :2] = zero_mean_unit_variance(self.X[:,:2])
        
    def collect_samples(self, indices, noise_std):
        # if true values is quite low, then noise term will dominate (instead, see if any scaled version makes sense here)
        y = self.Y[indices] + np.random.normal(0, noise_std, size=len(indices))
        return y

    def _setup_graph(self):
        self.graph = nx.Graph()
        # add all intersections as nodes
        for r in self.map.row_pass_indices:
            for c in self.map.obstacle_cols:
                self.graph.add_node((r,c))

        delta_x = self.map.stack_len + 1
        # x-axis is row or the first element of the tuple
        dx_dy = [(0,2), (0,-2), (delta_x, 0), (-delta_x, 0)]

        # add all edges
        for node in self.graph.nodes():
            for dx, dy in dx_dy:
                neighbor = (node[0] + dx, node[1] + dy)
                if is_valid_cell(neighbor, self.map.shape):
                    self.graph.add_edge(node, neighbor)

        # store a backup
        self.backup_graph = deepcopy(self.graph)

    def _pre_search(self, start, waypoints):
        new_nodes, new_edges, remove_edges = get_new_nodes_and_edges(self.graph, self.map, [start] + waypoints)
        self.graph.add_nodes_from(new_nodes)
        self.graph.add_edges_from(new_edges)
        # remove redundant edges (these edges have been replaced by multiple edges between waypoints and map junctions)
        self.graph.remove_edges_from(remove_edges)        

    def _post_search(self):
        self.graph = deepcopy(self.backup_graph)

    def get_shortest_path_waypoints(self, start, heading, waypoints, allowance=1):
        self._pre_search(start, waypoints)

        nw = len(waypoints)
        # expansion tree
        tree = nx.DiGraph()
        # tree_node contains a boolean vector representing which waypoints have been visited
        tree_start_node = (start, tuple([False]*nw), heading)
        tree.add_node(tree_start_node, pos=tree_start_node[0])
        
        open_list = [tree_start_node]
        gvals = [0]
        closed_list = []

        # an upper bound on the shortest path length (move to the nearest waypoint)
        shortest_length = upper_bound(start, heading, waypoints, self.graph, self.extra_cost)

        # for efficieny, it will be beneficial if nodes are expanded in increasing order of gval
        while len(open_list) > 0:
            tree_node = open_list.pop(0)
            node = tree_node[0]
            gval = gvals.pop(0)

            ngh = self.graph.neighbors(node)
            for new_node in ngh:
                cost = edge_cost(node, tree_node[2], new_node)
                if cost == np.inf:
                    continue
                new_gval = gval + cost

                new_heading = get_heading(new_node, node)
                new_tree_node_visited = deepcopy(tree_node[1])
                if new_node in waypoints:
                    new_tree_node_visited = list(new_tree_node_visited)
                    new_tree_node_visited[waypoints.index(new_node)] = True
                    new_tree_node_visited = tuple(new_tree_node_visited)
                new_tree_node = (new_node, new_tree_node_visited, new_heading)

                # this can be multiplied by an allowance factor to allow paths greater than shortest length
                to_go = distance_to_go(new_tree_node, waypoints)
                if new_gval + to_go > allowance * shortest_length:
                    continue
                
                action = node_action(tree, new_tree_node)
                if action == 'merge':
                    tree.add_edge(tree_node, new_tree_node)
                    continue
                # for informative paths which can be greater than the shortest path, it is not optimum to discard paths
                if allowance==1 and action == 'discard':
                    continue
            
                # add new node to tree
                tree.add_node(new_tree_node, pos=new_tree_node[0])
                tree.add_edge(tree_node, new_tree_node)

                if sum(new_tree_node_visited) == nw:
                    shortest_length = min(new_gval, shortest_length)
                    closed_list.append(new_tree_node)

                else:
                    open_list.append(new_tree_node)
                    gvals.append(new_gval)
                    
        all_paths_gen = [nx.all_shortest_paths(tree, tree_start_node, t) for t in closed_list]
        all_paths = []
        all_paths_indices = []
        all_paths_cost = []
        for path_gen in all_paths_gen:
            for i, path in enumerate(path_gen):
                locs = [p[0] for p in path]
                gp_indices = [self.gp_indices_between(locs[i],locs[i+1]) for i in range(len(path)-1)]
                # need to add the last sampling location 
                gp_indices = [item for sublist in gp_indices for item in sublist] + [self.map_pose_to_gp_index(locs[-1])]

                cost = path_cost(locs)
                # print(i, locs, cost)

                all_paths.append(locs)
                all_paths_indices.append(gp_indices)
                all_paths_cost.append(cost)

        self._post_search()
        return all_paths, all_paths_indices, all_paths_cost

    def get_path_from_checkpoints(self, checkpoints):
        # consecutive checkpoints are always aligned along either x-axis or y-axis
        path = [checkpoints[0]]
        for i in range(len(checkpoints)-1):
            heading = get_heading(checkpoints[i+1], path[-1])
            while path[-1]!=checkpoints[i+1]:
                path.append((path[-1][0] + heading[0], path[-1][1] + heading[1]))
        return path

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

    def gp_indices_between(self, map_pose0, map_pose1):
        # returns list of gp indices between map_pose0 and map_pose1 "excluding" map_pose1 location
        diff = (map_pose1[0]-map_pose0[0], map_pose1[1]-map_pose0[1])
        if diff[0] == 0:
            return []
        if diff[1] == 0:
            inc = diff[0]//abs(diff[0])
            indices = self.indices_mask[map_pose0[0]: map_pose1[0]: inc, map_pose0[1]]
            indices = [ind for ind in indices if ind is not None]
            return indices