import networkx as nx 
import numpy as np
import ipdb
from utils import manhattan_distance
import matplotlib.pyplot as plt
from pprint import pprint


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


def get_heading(node, previous_node):
    diff = (node[0]-previous_node[0], node[1]-previous_node[1])
    if diff[0]==0:
        return (0,diff[1]//abs(diff[1]))
    else:
        return (diff[0]//abs(diff[0]),0)


def get_new_nodes_and_edges(graph, map, nodes):
    # return nodes and edges to be added to the graph and the edges to be removed from the graph
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
    

def distance_to_go(node, waypoints):
    # remaining waypoints
    left = [w for i,w in enumerate(waypoints) if not node[1][i]]
    all_locs = np.vstack(left + [node[0]])
    # boundaries of the bounding box
    mins = np.min(all_locs, axis=0)
    maxs = np.max(all_locs, axis=0)
    t = np.array(node[0])
    a = t - mins
    b = maxs - t
    dist = sum(a) + sum(b) + min(a[0],b[0]) + min(a[1],b[1])
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
