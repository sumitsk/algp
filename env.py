import numpy as np
import ipdb

from map import Map
from utils import load_data, zero_mean_unit_variance, is_valid_cell, load_dataframe, normalize
from networkx import nx
from copy import deepcopy
from graph_utils import get_new_nodes_and_edges, edge_cost, get_heading, lower_bound_path_cost, find_merge_to_node

class FieldEnv(object):
    def __init__(self, num_rows=15, num_cols=15, data_file=None, phenotype='plant_count', num_test=40):
        super(FieldEnv, self).__init__()

        if data_file is None:
            from utils import generate_gaussian_data, generate_mixed_data
            self.num_rows, self.num_cols = num_rows, num_cols
            self.X, self.Y = generate_gaussian_data(num_rows, num_cols)
            # self.X, self.Y = generate_mixed_data(num_rows, num_cols)

        else:
            # self.num_rows, self.num_cols, self.X, self.Y = load_data(data_file)
            self.num_rows, self.num_cols, self.X, self.Y, self.category = load_dataframe(data_file, 
                                                                          target_feature=phenotype,
                                                                          extra_input_features=['grvi', 'leaf_fill'])
            # NOTE: self.X should contain samples row-wise

            # take out a category-wise test set scattered uniformly across genotypes
            test_inds = []
            for i in range(4):
                ind = np.random.choice(np.where(self.category==i)[0], num_test//4, replace=False)
                test_inds.append(ind)
            test_inds = np.hstack(test_inds)
            train_inds = np.array(list(set(range(len(self.X))) - set(test_inds)))
            
            self.test_X = np.copy(self.X[test_inds])
            self.test_Y = np.copy(self.Y[test_inds])
            self.test_category = np.copy(self.category[test_inds])
            
            self.X = self.X[train_inds]
            self.Y = self.Y[train_inds]
            self.category = self.category[train_inds]

        self.map = Map(self.num_rows, self.num_cols)
        self._place_samples()

        # self._normalize_dataset()
        self._setup_graph()

    def _place_samples(self):
        self.map_pose_to_gp_index_matrix = np.full(self.map.shape, None)
        self.gp_index_to_map_pose_array = np.full(len(self.X), None)
        row = 0
        x = self.X[:,:2]
        indices = np.arange(len(x))
        for i in range(self.map.shape[0]):
            if i in self.map.row_pass_indices:
                continue
            row += 2
            row_indices = indices[x[:,0]==row]
            for ind in row_indices:
                map_pose = (i, int(2*x[ind,1] - 2))
                self.map_pose_to_gp_index_matrix[map_pose] = ind 
                self.gp_index_to_map_pose_array[ind] = map_pose
        
    def vec_to_gp_mat(self, vec, default_value=0.0):
        arr = np.full(self.map.shape, default_value)
        for i,v in enumerate(vec):
            arr[self.gp_index_to_map_pose_array[i]] = v
        arr = np.delete(arr, self.map.row_pass_indices, axis=0)
        arr = np.delete(arr, self.map.obstacle_cols, axis=1)
        return arr

    def _normalize_dataset(self):
        # NOTE: this is fine since we know in advance about the samples
        self.X = self.X.astype(float)
        self.X[:, :4] = zero_mean_unit_variance(self.X[:,:4])
        # self.Y = normalize(self.Y)
        
    def collect_samples(self, indices, noise_std):
        y = self.Y[indices] + np.random.normal(0, noise_std, size=len(indices))
        return y

    def _setup_graph(self):
        self.graph = nx.Graph()
        # add all intersections as nodes
        for r in self.map.row_pass_indices:
            for c in self.map.free_cols:
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
        # add start and waypoints to the graph and remove redundant ones
        new_nodes, new_edges, remove_edges = get_new_nodes_and_edges(self.graph, self.map, [start] + waypoints)
        self.graph.add_nodes_from(new_nodes)
        self.graph.add_edges_from(new_edges)
        # remove redundant edges (these edges have been replaced by edges between waypoints and map junctions)
        self.graph.remove_edges_from(remove_edges)        

    def _post_search(self):
        self.graph = deepcopy(self.backup_graph)

    def get_shortest_path_waypoints(self, start, heading, waypoints, beta=1):
        self._pre_search(start, waypoints)

        # start_time = time.time()
        nw = len(waypoints)
        # expansion tree
        tree = nx.DiGraph()
        # node attributes = {pos, gval, visited, heading}

        root = 0
        tree.add_node(root, pose=start, heading=heading, visited=[False]*nw, gval=0)
        open_list = [root]
        closed_list = []

        # an upper bound on the shortest path length (always moving to the nearest waypoint)
        # shortest_length = self.map.nearest_waypoint_path_cost(start, heading, waypoints)
        shortest_length = self.bnb_shortest_path(start, heading, waypoints)

        # for efficieny, it will be beneficial if nodes are expanded in increasing order of gval
        idx = root
        count_merged = 0
        count_skipped = 0
        while len(open_list) > 0:
            parent_idx = open_list.pop(0)
            tree_node = tree.node[parent_idx]
            pose = tree_node['pose']
            gval = tree_node['gval']

            ngh = self.graph.neighbors(pose)
            for new_pose in ngh:
                cost = edge_cost(pose, tree_node['heading'], new_pose)
                # can't move back to its parent node (or can't take a u-turn)
                if cost == np.inf:
                    continue
                new_gval = gval + cost

                new_heading = get_heading(new_pose, pose)
                new_visited = deepcopy(tree_node['visited'])
                if new_pose in waypoints:
                    new_visited[waypoints.index(new_pose)] = True

                remaining_waypoints = [w for i,w in enumerate(waypoints) if not new_visited[i]]
                # min_dist_to_go = lower_bound_path_cost(new_pose, remaining_waypoints)
                min_dist_to_go = self.bnb_shortest_path(new_pose, new_heading, remaining_waypoints, shortest_length)
                if new_gval + min_dist_to_go > beta * shortest_length:
                    continue
                
                new_tree_node = dict(pose=new_pose, heading=new_heading, visited=new_visited, gval=new_gval)
                merge_to = find_merge_to_node(tree, new_tree_node)
                if merge_to is not None:
                    tree.add_edge(parent_idx, merge_to, weight=cost)
                    count_merged += 1
                    continue
                
                # NOTE: because of gp_indices computation, this is slow
                # action = self.node_action(tree, new_tree_node, tree_node, parent_idx)
                # if action == 'continue':
                #     count_skipped += 1
                #     continue

                # if action is not None:
                #     tree.add_edge(parent_idx, action, weight=cost)
                #     count_merged += 1
                #     continue
                    
                # add new node to tree
                idx = idx + 1
                tree.add_node(idx, **new_tree_node)
                tree.add_edge(parent_idx, idx, weight=cost)

                if sum(new_visited) == nw:
                    shortest_length = min(new_gval, shortest_length)
                    closed_list.append(idx)
                else:
                    open_list.append(idx)
        # end_time = time.time()
        # print('Time {:4f}'.format(end_time-start_time))
        
        # start_time = time.time()
        all_paths_gen = [nx.all_shortest_paths(tree, root, t, weight='weight') for t in closed_list]
        # end_time = time.time()
        # print('Time {:4f}'.format(end_time-start_time))

        # start_time = time.time()
        all_paths = []
        all_paths_indices = []
        all_paths_cost = []
        for path_gen in all_paths_gen:
            for i, path in enumerate(path_gen):
                locs = [tree.node[p]['pose'] for p in path]
                
                # this step is computationally expensive (determining indices on a path)
                gp_indices = [self.gp_indices_between(locs[t],locs[t+1]) for t in range(len(path)-1)]
                # need to add the last sampling location 
                gp_indices = [item for sublist in gp_indices for item in sublist] + [self.map_pose_to_gp_index(locs[-1])]
                all_paths_indices.append(gp_indices)
                
                path_cost = tree.node[path[-1]]['gval']
                all_paths.append(locs)
                all_paths_cost.append(path_cost)

        # end_time = time.time()
        # print('Time {:4f}'.format(end_time-start_time))
        # visualize tree 
        # import matplotlib.pyplot as plt
        # pos = nx.get_node_attributes(tree, 'pose')
        # nx.draw_networkx(tree, pos)
        # plt.show()

        # print(count_merged)
        # print(count_skipped)
        # print(len(all_paths))
        self._post_search()
        return all_paths, all_paths_indices, all_paths_cost

    def gp_indices_on_path(self, path):
        gp_indices = [self.gp_indices_between(path[t],path[t+1]) for t in range(len(path)-1)]
        gp_indices = [item for sublist in gp_indices for item in sublist]        
        return gp_indices

    def node_action(self, tree, node, parent_node, parent_idx):
        # all nodes in the graph with same attributes as node
        all_idx = [n for n in tree.nodes() if tree.node[n]==node]
        if len(all_idx) > 0:
            assert len(all_idx)==1, 'More than one path found!!!'
            sim_idx = all_idx[0]
            # find gp indices along sim_node
            gen = nx.all_shortest_paths(tree, 0, sim_idx, weight='weight')
            sim_path = [p for p in gen]
            sim_locs = [tree.node[p]['pose'] for p in sim_path[0]]
            sim_gp_ind = self.gp_indices_on_path(sim_locs)
            
            gen = nx.all_shortest_paths(tree, 0, parent_idx)
            parent_path = [p for p in gen]
            locs = [tree.node[p]['pose'] for p in parent_path[0]] + [node['pose']]
            gp_ind = self.gp_indices_on_path(locs)
            
            # do not add node to tree
            if set(gp_ind).issubset(set(sim_gp_ind)):
                return 'continue'
            return all_idx[0]
        return None  

    def get_path_from_checkpoints(self, checkpoints):
        # consecutive checkpoints are always aligned along either x-axis or y-axis
        path = [checkpoints[0]]
        for i in range(len(checkpoints)-1):
            heading = get_heading(checkpoints[i+1], path[-1])
            while path[-1]!=checkpoints[i+1]:
                path.append((path[-1][0] + heading[0], path[-1][1] + heading[1]))
        return path

    def bnb_shortest_path(self, start, heading, waypoints, least_cost=None):
        # do branch and bound search to find shortest path
        if len(waypoints) == 0:
            return 0

        if least_cost is None:
            least_cost = self.map.nearest_waypoint_path_cost(start, heading, waypoints)
        
        nw = len(waypoints)
        tree = nx.DiGraph()
        # node attributes = {pos, gval, visited, heading}

        root = 0
        tree.add_node(root, pose=start, heading=heading, visited=[False]*nw, gval=0)
        open_list = [root]
        closed_list = []
        idx = root

        while len(open_list) > 0:
            parent_idx = open_list.pop(0)
            parent_node = tree.node[parent_idx]
            
            # neighbors are all waypoints which haven't been visited yet
            for i in range(nw):
                if parent_node['visited'][i]:
                    continue
                cost, final_heading = self.map.distance_between_nodes(parent_node['pose'], waypoints[i], parent_node['heading'])
                new_gval = parent_node['gval'] + cost
                if new_gval > least_cost:
                    continue

                new_visited = np.copy(parent_node['visited'])
                new_visited[i] = True
                child_node = dict(pose=waypoints[i], heading=final_heading, visited=new_visited, gval=new_gval)
                
                idx += 1
                tree.add_node(idx, **child_node)
                tree.add_edge(parent_idx, idx, weight=cost)
                if sum(new_visited) == nw:
                    least_cost = min(new_gval, least_cost)
                    closed_list.append(idx)
                else:
                    open_list.append(idx)
        return least_cost

    @property
    def shape(self):
        return self.num_rows, self.num_cols

    @property
    def num_samples(self):
        return self.X.shape[0]

    def gp_indices_between(self, map_pose0, map_pose1):
        # returns list of gp indices between map_pose0 and map_pose1 "excluding" map_pose1 location
        diff = (map_pose1[0]-map_pose0[0], map_pose1[1]-map_pose0[1])
        if diff[0] == 0:
            return []
        if diff[1] == 0:
            inc = diff[0]//abs(diff[0])
            indices = self.map_pose_to_gp_index_matrix[map_pose0[0]: map_pose1[0]: inc, map_pose0[1]]
            indices = [ind for ind in indices if ind is not None]
            return indices

    def gp_index_to_map_pose(self, gp_index):
        return self.gp_index_to_map_pose_array[gp_index]

    def map_pose_to_gp_index(self, map_pose):
        assert isinstance(map_pose, tuple), 'Map pose must be a tuple'
        return self.map_pose_to_gp_index_matrix[map_pose]


if __name__ == '__main__':
    env = FieldEnv(data_file='data/female_gene_data/all_mean.pkl')
    pose = (17,54)
    heading = (1,0)
    waypoints = [(1,22), (16,12), (11,38), (15,52), (3,68)]
    least_cost = env.bnb_shortest_path(pose, heading, waypoints)

    import time

    start = time.time()
    paths, indices, costs = env.get_shortest_path_waypoints(pose, heading, waypoints, beta=1)
    end = time.time()
    print('Time consumed: {:4f}'.format(end-start))