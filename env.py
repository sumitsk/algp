import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from networkx import nx
from copy import deepcopy

from map import Map
from utils import is_valid_cell, load_data_from_pickle, draw_path, manhattan_distance, generate_phenotype_data
from graph_utils import get_down_and_up_nodes, edge_cost, get_heading, find_merge_to_node, lower_bound_path_cost
          
import ipdb


class FieldEnv(object):
    # grid-based simulation environment 
    def __init__(self, data_file=None, phenotype='plant_count', num_test=40):
        super(FieldEnv, self).__init__()
        if data_file is None:
            self.num_rows = 30
            self.num_cols = 30
            x, y, self.y_category = generate_phenotype_data(num_rows=self.num_rows, num_cols=self.num_cols, num_zs=4)
            x[:,1] *= 2
            self._setup(x, y, num_test)
            self._place_samples_others(row_start=0, row_inc=1)

        else:
            # NOTE: will deprecate this soon
            # for intel dataset
            if 'intel' in data_file:
                import scipy.io
                mat = scipy.io.loadmat(data_file)
                x = mat['Xss']
                y = mat['Fss'].squeeze()
                self.num_rows = 15
                self.num_cols = 17                
                self._setup(x, y, num_test)
                self._place_samples_others(row_start=0, row_inc=3)

            # for sorghum dataset
            else: 
                extra_features = ['leaf_fill', 'grvi']
                max_range = 35
                self.num_rows, self.num_cols, x, y = load_data_from_pickle(data_file, target_feature=phenotype,
                                                                           max_range=max_range, 
                                                                           extra_input_features=extra_features)
                self._setup(x, y, num_test)
                self._place_samples_pheno()

        self.all_x = np.copy(x)
        self.all_y = np.copy(y)
        
        self._setup_graph()
        # for rendering
        self.fig = None
        self.ax = None

    def _setup(self, x, y, num_test):
        # split into training and testing data
        n = len(x)
        perm = np.random.permutation(n)
        test_ind = perm[:num_test]
        train_ind = perm[num_test:]
        
        self.X = x[train_ind]
        self.Y = y[train_ind]
        self.test_X = x[test_ind]
        self.test_Y = y[test_ind]

        # setup map and pose-index and index-pose lookup tables
        self.map = Map(self.num_rows, self.num_cols, num_row_passes=4)
        self.map_pose_to_gp_index_matrix = np.full(self.map.shape, None)
        self.gp_index_to_map_pose_array = np.full(len(self.X), None)

    # TODO: merge this with the next function
    def _place_samples_others(self, row_start=0, row_inc=1):
        # assign samples to grid cells
        x = self.X[:,:2]
        indices = np.arange(len(x))
        row = row_start
        for i in range(self.map.shape[0]):
            if i in self.map.row_pass_indices:
                continue
            row_indices = indices[x[:,0]==row]
            for ind in row_indices:
                map_pose = (i, int(x[ind,1]))
                self.map_pose_to_gp_index_matrix[map_pose] = ind 
                self.gp_index_to_map_pose_array[ind] = map_pose
            row += row_inc

    def _place_samples_pheno(self):
        # assign samples to grid cells
        row = 2
        row_inc = 2
        x = self.X[:,:2]
        indices = np.arange(len(x))
        for i in range(self.map.shape[1]):
            if i%2 == 1:
                continue
            row_indices = indices[x[:,0]==row]
            for ind in row_indices:
                t = x[ind, 1] + (x[ind, 1] -1) // self.map.corridor_len
                map_pose = (int(t), row-2)
                self.map_pose_to_gp_index_matrix[map_pose] = ind 
                self.gp_index_to_map_pose_array[ind] = map_pose
            row += row_inc
            
    def collect_samples(self, indices, noise_std):
        # draw measurement for the given sampling index and noise
        y = self.Y[indices] + np.random.normal(0, noise_std)
        # truncating negative values to 0
        y = max(0,y)
        return y

    def _setup_graph(self):
        # initialize graph for path planning
        self.graph = nx.Graph()
        # add all intersections as nodes
        for r in self.map.row_pass_indices:
            for c in self.map.free_cols:
                self.graph.add_node((r,c), pose=(c,self.map.shape[0]-r), new='False')

        delta_x = self.map.corridor_len + 1
        # x-axis is row or the first element of the tuple
        dx_dy = [(0,2), (0,-2), (delta_x, 0), (-delta_x, 0)]

        # add all edges
        for node in self.graph.nodes():
            for dx, dy in dx_dy:
                neighbor = (node[0] + dx, node[1] + dy)
                if is_valid_cell(neighbor, self.map.shape):
                    indices = self.gp_indices_between(node, neighbor)
                    self.graph.add_edge(node, neighbor, indices=indices)

        # store a backup graph (or restore point)
        self.backup_graph = deepcopy(self.graph)

    def _pre_search(self, start, waypoints):
        # nodes and edges to be added to the graph and the edges to be removed from the graph
        new_nodes, new_edges, new_edges_indices, remove_edges = self.get_new_nodes_and_edges([start] + waypoints)
        
        # add start and waypoint nodes to the graph
        for node in new_nodes:
            self.graph.add_node(node, pose=(node[1],self.map.shape[0]-node[0]), new='True')
        
        # add edges
        for edge, indices in zip(new_edges, new_edges_indices):
            self.graph.add_edge(edge[0], edge[1], indices=indices)    
        
        # remove redundant edges (these edges have been replaced by edges between waypoints and map junctions)
        self.graph.remove_edges_from(remove_edges)  

        # draw graph
        # colors = []
        # for n in self.graph:
        #     if self.graph.node[n]['new'] == 'False':
        #         colors.append('darkorange')
        #     else:
        #         colors.append('green')
        # pose = nx.get_node_attributes(self.graph, 'pose')
        # nx.draw(self.graph, pose, node_color=colors, width=5)
        # plt.show()
        
    def get_new_nodes_and_edges(self, new_nodes):
        # return nodes and edges to be added to the graph and the edges to be removed from the graph
        
        # nodes not present in the graph already
        new_nodes = [n for n in new_nodes if n not in self.graph.nodes()]
        new_edges = []
        new_edges_indices = []
        remove_edges = []

        # for each node find edges to be added and to be removed from the graph
        for node in new_nodes:
            down_junc = self.map.get_down_junction(node)
            up_junc = self.map.get_up_junction(node)
            down_node, up_node = get_down_and_up_nodes(node, new_nodes, down_junc, up_junc)
            
            down_indices = self.gp_indices_between(down_node, node)
            if self.map_pose_to_gp_index_matrix[down_node] is not None:
                down_indices.pop(0)
            up_indices = self.gp_indices_between(up_node, node)
            if self.map_pose_to_gp_index_matrix[up_node] is not None:
                up_indices.pop(0)
            
            new_edges.append((down_node, node))
            new_edges.append((node, up_node))
            new_edges_indices.append(down_indices)
            new_edges_indices.append(up_indices)

            remove_edges.append((down_junc, up_junc))   
        return new_nodes, new_edges, new_edges_indices, remove_edges

    def _post_search(self):
        self.graph = deepcopy(self.backup_graph)

    def get_all_paths(self, start, heading, waypoints, heuristic_cost=None, slack=0):
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

        least_cost = self.get_heuristic_cost(start, heading, waypoints) if heuristic_cost is None else heuristic_cost

        # for efficieny, it will be beneficial if nodes are expanded in increasing order of gval
        idx = root
        count_merged = 0
        # count_skipped = 0
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

                new_heading = get_heading(pose, new_pose)
                new_visited = deepcopy(tree_node['visited'])
                if new_pose in waypoints:
                    new_visited[waypoints.index(new_pose)] = True

                remaining_waypoints = [w for i,w in enumerate(waypoints) if not new_visited[i]]
                # min_dist_to_go = self.get_heuristic_cost(new_pose, new_heading, remaining_waypoints, least_cost)
                min_dist_to_go = lower_bound_path_cost(new_pose, remaining_waypoints)
                if new_gval + min_dist_to_go > least_cost + slack:
                    # print('Skipping!')
                    continue
                
                new_tree_node = dict(pose=new_pose, heading=new_heading, visited=new_visited, gval=new_gval)
                merge_to = find_merge_to_node(tree, new_tree_node)
                if merge_to is not None:
                    tree.add_edge(parent_idx, merge_to, weight=cost)
                    count_merged += 1
                    # print('Merging')
                    continue
                
                # NOTE: because of gp_indices computation, this is slow
                # with the new method, this can be uncommented for further tree pruning
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
                    least_cost = min(new_gval, least_cost)
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
                path_cost = tree.node[path[-1]]['gval']
                if path_cost > least_cost + slack:
                    continue

                all_paths_cost.append(path_cost)
                locs = [tree.node[p]['pose'] for p in path]
                # gp_indices contains only mobile sensing locations
                gp_indices = [self.graph.get_edge_data(locs[t], locs[t+1])['indices'] for t in range(len(locs) - 1)]
                # gp_indices = [self.gp_indices_between(locs[t],locs[t+1]) for t in range(len(path)-1)]
                
                gp_indices = [item for sublist in gp_indices for item in sublist]
                all_paths_indices.append(gp_indices)
                
                all_paths.append(locs)
                
        # end_time = time.time()
        # print('Time {:4f}'.format(end_time-start_time))

        # print(count_merged)
        # print(count_skipped)
        # print(len(all_paths))
        self._post_search()
        return all_paths, all_paths_indices, all_paths_cost

    def get_heuristic_cost(self, start, heading, waypoints, least_cost_ub=None, return_seq=False):
        if len(waypoints) == 0:
            return 0

        least_cost = self.map.nearest_waypoint_path_cost(start, heading, waypoints) if least_cost_ub is None else least_cost_ub
        
        gval = 0
        if start[0] not in self.map.row_pass_indices:
            if heading not in [(1,0), (-1,0)]:
                raise ValueError('Impossible setting encountered!!')
        else:
            # move to the first junction 
            junc = self.map.get_junction(start, heading)
            x = start[0]
            covered = [False]*len(waypoints)
            costs = [-1]*len(waypoints)
            while x != junc[0]:
                pose = (x,start[1])
                if pose in waypoints:
                    itr = waypoints.index(pose)
                    covered[itr] = True
                    costs[itr] = abs(start[0] - x)
                x = x + heading[0]
            waypoints = [w for i,w in enumerate(waypoints) if not covered[i]]
            if len(waypoints) == 0:
                return max(costs)
            gval = manhattan_distance(start, junc)
            start = junc
        
        nw = len(waypoints)
        tree = nx.DiGraph()
        # node attributes = {pos, gval, visited, heading}

        root = 0
        tree.add_node(root, pose=start, heading=heading, visited=[False]*nw, gval=gval)
        open_list = [root]
        closed_list = []
        idx = root

        while len(open_list) > 0:
            parent_idx = open_list.pop(0)
            parent_node = tree.node[parent_idx]
            
            # neighbors are all the waypoints which haven't been visited yet
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
                    if new_gval <= least_cost:
                        least_cost = new_gval
                        best_idx = idx

                    closed_list.append(idx)
                else:
                    open_list.append(idx)
        if return_seq:
            ipdb.set_trace()
        return least_cost

    def gp_indices_on_path(self, path):
        # all gp indices lying on the path
        gp_indices = [self.graph.get_edge_data(path[t], path[t+1])['indices'] for t in range(len(path) - 1)]
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
            heading = get_heading(path[-1], checkpoints[i+1])
            while path[-1]!=checkpoints[i+1]:
                path.append((path[-1][0] + heading[0], path[-1][1] + heading[1]))
        return path

    @property
    def shape(self):
        return self.num_rows, self.num_cols

    @property
    def num_samples(self):
        return len(self.X)

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

    def render_map(self, ax, next_path_waypoints, all_paths, next_static_locations, all_static_locations):
        # ax.set_title('Environment')
        sample_color = np.array([255,218,185])/255
        plot = 1.0 - np.repeat(self.map.occupied[:, :, np.newaxis], 3, axis=2)
        for i in range(plot.shape[0]):
            for j in range(plot.shape[1]):
                if self.map_pose_to_gp_index_matrix[i,j] is not None:
                    plot[i,j] = sample_color
    
        all_paths_color = np.array([244,164,96])/255
        all_static_locations_color = np.array([127, 255, 0])/255
        next_static_locations_color = np.array([0, 100, 0])/255
        pose_color = np.array([1,0,0])
        
        # highlight all mobile samples locations 
        if len(all_paths) > 0:
            plot[all_paths[:, 0], all_paths[:, 1], :] = all_paths_color
        
        # highlight all static samples locations
        if len(all_static_locations) > 0:
            plot[all_static_locations[:, 0], all_static_locations[:, 1], :] = all_static_locations_color

        # highlight next static samples
        if len(next_static_locations) > 0:
            plot[next_static_locations[:,0], next_static_locations[:,1], :] = next_static_locations_color

        # show the next path
        waypoints = [x[::-1] for x in next_path_waypoints]
        draw_path(ax, waypoints, head_width=0.25, head_length=.2, linewidth=3.0, delta=None, color='green')

        # highlight robot pose
        pose = next_path_waypoints[0]
        plot[pose[0], pose[1], :] = pose_color
        ax.imshow(plot)
        # sns.heatmap(plot, ax=ax, cbar=False)

    def render(self, next_path_waypoints, all_paths, next_static_locations, all_static_locations, true=None, pred=None):
        if true is not None and pred is not None:
            num_axes = 3
        else:
            num_axes = 1

        self._setup_render(num_axes=num_axes)

        self.render_map(self.ax[0], next_path_waypoints, all_paths, next_static_locations, all_static_locations)
        if num_axes == 3:
            sz = 15
            self.ax[0].set_title('Field Environment', fontsize=sz)
            self.ax[1].set_title('Predicted Phenotype distribution', fontsize=sz)
            self.ax[2].set_title('Actual Phenotype distribution', fontsize=sz)
            vmin = true.min()
            vmax = true.max()
            # TODO: colorbar is not cleared when .cla() is called
            sns.heatmap(pred, ax=self.ax[1], cmap='ocean', vmin=vmin, vmax=vmax, cbar=False, square=True)
            sns.heatmap(true, ax=self.ax[2], cmap='ocean', vmin=vmin, vmax=vmax, cbar=False, square=True)
        plt.pause(1)

    def _setup_render(self, num_axes):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(ncols=num_axes, figsize=(4*num_axes, 4))
            self.ax = [self.ax] if num_axes==1 else self.ax.flatten()
            # clear all ticks
            for ax_ in self.ax:
                ax_.get_xaxis().set_visible(False)
                ax_.get_yaxis().set_visible(False)
        else:
            # clear all axes
            for ax_ in self.ax:
                ax_.cla()

    def render_naive(self):
        num_axes = 1
        self._setup_render(num_axes)

        # waypoints
        first_array = [(i,0) for i in range(self.map.shape[0])]
        second_array = [(i,2) for i in reversed(range(self.map.shape[0]))]
        third_array = [(i,4) for i in range(self.map.shape[0])]
        

        all_paths = []
        next_path_waypoints = first_array + [(self.map.shape[0]-1,1)] + second_array + [(0,3)] + third_array
        next_static_locations = []
        all_static_locations = []
        self.render_map(self.ax[0], next_path_waypoints, all_paths, next_static_locations, all_static_locations)
        plt.pause(1)
        ipdb.set_trace()



