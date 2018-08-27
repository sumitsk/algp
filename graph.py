import networkx as nx 
from env import FieldEnv
import numpy as np
import ipdb
from utils import is_valid_cell, manhattan_distance
import matplotlib.pyplot as plt
from copy import deepcopy
from pprint import pprint


# heading should be a tuple
def edge_cost(node, heading, next_node):
    # heading must be one from {(0,1), (0,-1), (1,0), (-1,0)}
    diff = (next_node[0]-node[0], next_node[1]-node[1])
    if heading[0]==0:
        if diff[1]*heading[1] < 0:
            return np.inf
    else:
        if diff[0]*heading[0] < 0:
            return np.inf
    return abs(diff[0]) + abs(diff[1])


def shortest_path(start, heading, end, graph, map):
    # returns shortest path between start and end
    # assuming agent can't take a u-turn 
    
    new_nodes, new_edges, remove_edges = get_new_nodes_and_edges(graph, map, [start, end])
    for node in new_nodes:
        graph.add_node(node, pos=node)
    graph.add_edges_from(new_edges)
    # remove redundant edges (these edges have been replaced by multiple edges between waypoints and map junctions)
    graph.remove_edges_from(remove_edges)

    open_list = [start]
    gvals = [0]
    headings = [heading]
    
    # expansion tree
    tree = nx.Graph()
    tree.add_node(start, pos=start)
    shortest_length = np.inf

    while len(open_list) > 0:
        node = open_list.pop(0)
        h = headings.pop(0)
        gval = gvals.pop(0)

        ngh = graph.neighbors(node)
        for new_node in ngh:
            if new_node in tree.nodes():
                continue
            
            new_gval = gval + edge_cost(node, h, new_node)
            if new_gval >= shortest_length:
                continue
            open_list.append(new_node)
            gvals.append(new_gval)
            new_heading = get_heading(new_node, node)
            headings.append(new_heading)

            # add new node to tree
            tree.add_node(new_node, pos=new_node)
            tree.add_edge(node, new_node)

            if new_node == end:
                shortest_length = min(new_gval, shortest_length)                

    # TODO: shortest path can be estimated by backtracking from the end node, better: develop a tree structure
    print(shortest_length)


def get_heading(node, previous_node):
    diff = (node[0]-previous_node[0], node[1]-previous_node[1])
    # print(diff)
    if diff[0]==0:
        return (0,diff[1]//abs(diff[1]))
    else:
        return (diff[0]//abs(diff[0]),0)


def get_new_nodes_and_edges(graph, map, nodes):
    # return nodes and edges to be added to the graph
    new_nodes = [n for n in nodes if n not in graph.nodes()]
    new_edges = []
    remove_edges = []
    for node in new_nodes:
        lower_junc, upper_junc = get_junctions(map, node)
        lower, upper = get_lower_and_upper_nodes(node, new_nodes, lower_junc, upper_junc)
        new_edges.append((lower, node))
        new_edges.append((node, upper))
        remove_edges.append((lower_junc, upper_junc))   
    return new_nodes, new_edges, remove_edges


def get_lower_and_upper_nodes(node, others, lower_junc, upper_junc):
    # find nodes just above and below the current node
    lower, upper = lower_junc, upper_junc
    for n in others:
        if in_between(n, lower, node):
            lower = n
        if in_between(n, node, upper):
            upper = n
    return lower, upper


def in_between(node, lower, upper):
    # check if node lies between lower and upper
    if node[1] == lower[1] and node[1] == upper[1]:
        if lower[0] < node[0] < upper[0]:
            return True
    return False


def get_junctions(map, node):
    # return junctions just upper and lower the node in the map
    lower = max([x for x in map.row_pass_indices if x<node[0]])
    upper = min([x for x in map.row_pass_indices if x>node[0]])
    return (lower, node[1]), (upper, node[1])


def get_shortest_path_waypoints(start, heading, waypoints, graph, map, allowance):
    new_nodes, new_edges, remove_edges = get_new_nodes_and_edges(graph, map, [start] + waypoints)
    graph.add_nodes_from(new_nodes)
    graph.add_edges_from(new_edges)
    # remove redundant edges (these edges have been replaced by multiple edges between waypoints and map junctions)
    graph.remove_edges_from(remove_edges)

    nw = len(waypoints)
    # expansion tree
    tree = nx.DiGraph()
    # tree_node contains a boolean vector representing which waypoints have been visited
    tree_start_node = (start, tuple([False]*nw), heading)
    tree.add_node(tree_start_node, pos=tree_start_node[0])
    
    open_list = [tree_start_node]
    gvals = [0]
    closed_list = []

    extra_cost = 2*(map.stack_len + 3)
    shortest_length = upper_bound(start, heading, waypoints, graph, extra_cost)

    # for efficieny, it will be beneficial if nodes are expanded in increasing order of gval
    while len(open_list) > 0:
        tree_node = open_list.pop(0)
        node = tree_node[0]
        gval = gvals.pop(0)
        # print(gval, shortest_length)

        ngh = graph.neighbors(node)
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

            # TODO: add a heurisitc distance-to-go
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
    # for path_gen in all_paths_gen:
    #     for i, path in enumerate(path_gen):
    #         locs = [p[0] for p in path]
    #         print(i, locs, path_cost(locs))
    #     print()
    ipdb.set_trace()
    

def distance_to_go(node, waypoints):
    # remaining waypoints
    left = [w for i,w in enumerate(waypoints) if not node[1][i]]
    all_locs = np.vstack(left + [node[0]])
    mins = np.min(all_locs, axis=0)
    maxs = np.max(all_locs, axis=0)
    t = np.array(node[0])
    a = t - mins
    b = maxs - t
    dist = sum(a) + sum(b) + min(a[0],b[0]) + min(a[1],b[1])
    # ipdb.set_trace()
    return dist


def opposite_headings(h0, h1):
    temp = h0[0]*h1[0] + h0[1]*h1[1]
    if temp==-1:
        return True
    return False


def upper_bound(start, start_heading, waypoints, graph, extra_cost):
    nw = len(waypoints)
    visited = [False]*nw
    total_cost = 0

    node = start
    heading = start_heading

    while sum(visited) != nw:
        all_min_costs = np.full(nw, np.inf)
        final_headings = [None]*nw
        # find shortest path from node to all the unvisited waypoints and greedily choose the nearest one
        for i in range(nw):
            if visited[i]:
                continue
            # minimum cost of all paths from node to the i^{th} waypoint
            path_gen = nx.all_shortest_paths(graph, node, waypoints[i])
            paths_costs = []
            paths_final_heading = []
            for path in path_gen:
                initial_heading = get_heading(path[1], path[0])
                final_heading = get_heading(path[-1], path[-2])
                cost = path_cost(path)
                # if initial heading is opposite of the current heading, then add extra cost
                if opposite_headings(initial_heading, heading):
                    first_edge_length = manhattan_distance(path[0], path[1])
                    cost += extra_cost - first_edge_length

                paths_costs.append(cost)
                paths_final_heading.append(final_heading)    

            idx = np.argmin(paths_costs)
            final_headings[i] = paths_final_heading[idx] 
            # min cost from node to waypoints[i]   
            all_min_costs[i] = paths_costs[idx]

        # select the nearest waypoint as the next node
        idx = np.argmin(all_min_costs)
        visited[idx] = True
        node = waypoints[idx]
        heading = final_headings[idx]
        total_cost += all_min_costs[idx]
    return total_cost


def path_cost(path):
    cost = 0
    for i in range(len(path)-1):
        cost += manhattan_distance(path[i], path[i+1])
    return cost


def node_action(tree, node):
    # determines what to do with the given node

    # all nodes in the graph with same position and orientation
    visiteds = [n[1] for n in tree.nodes() if n[0]==node[0] and n[2]==node[2]]
    for v in visiteds:
        diff = np.array(v)*1 - np.array(node[1])*1
        # if same node exists, then merge this node
        if (diff==0).all():
            return 'merge'
        # if better node exists then this node need not be added if max path length = shortest path length
        if (diff>=0).all() and diff.max()==1:
            return 'discard'
    return 'keep'
    

if __name__ == '__main__':  
    env = FieldEnv(data_file=None, num_rows=15, num_cols=37)
    # env.map.render()

    rows = env.map.row_pass_indices
    cols = np.arange(0, env.map.shape[1], 2)
    g = nx.Graph()
    for r in rows:
        for c in cols:
            g.add_node((r,c))

    delta_x = env.map.stack_len + 1
    # x-axis is row or the first element of the tuple
    dxdy = [(delta_x,0), (-delta_x,0), (0,2), (0,-2)]

    for node in g.nodes():
        for dx,dy in dxdy:
            neighbor = (node[0] + dx, node[1] + dy)
            if is_valid_cell(neighbor, env.map.shape):
                g.add_edge(node, neighbor)

    # heading = (1,0)
    # nodes = [(0,0), (5,4), (6,4), (7,4)]
    # new_nodes, new_edges = get_new_nodes_and_edges(g, env.map, nodes)

    # start = (6,4)
    # heading = (1,0)
    # end = (1,4)
    # shortest_path(start, heading, end, g, env.map)

    start = (0,0)
    waypoints = [(6,28), (10,16), (18,8), (1,8), (3, 30)]
    # waypoints = [(5,2), (2,6)]
    heading = (1,0)
    get_shortest_path_waypoints(start, heading, waypoints, g, env.map, allowance=1)  
        