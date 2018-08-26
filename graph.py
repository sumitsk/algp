import networkx as nx 
from env import FieldEnv
import numpy as np
import ipdb
from utils import is_valid_cell
import matplotlib.pyplot as plt
from copy import deepcopy


# heading should be a tuple
def get_cost(node, heading, next_node):
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
    # remove redundant edges (these edges have been removed by multiple edges between waypoints and map junctions)
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
            
            new_gval = gval + get_cost(node, h, new_node)
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


def get_shortest_path_waypoints(start, heading, waypoints, graph, map):
    new_nodes, new_edges, remove_edges = get_new_nodes_and_edges(graph, map, [start] + waypoints)
    for node in new_nodes:
        graph.add_node(node, pos=node)
    graph.add_edges_from(new_edges)
    # remove redundant edges (these edges have been removed by multiple edges between waypoints and map junctions)
    graph.remove_edges_from(remove_edges)

    nw = len(waypoints)
    # expansion tree
    tree = nx.Graph()
    # tree_node contains a boolean vector representing which waypoints have been visited
    tree_start_node = (start, tuple([False for _ in range(nw)]))
    tree.add_node(tree_start_node, pos=tree_start_node[0])
    shortest_length = np.inf

    open_list = [tree_start_node]
    gvals = [0]
    headings = [heading]

    while len(open_list) > 0:
        tree_node = open_list.pop(0)
        node = tree_node[0]
        h = headings.pop(0)
        gval = gvals.pop(0)

        ngh = graph.neighbors(node)
        for new_node in ngh:
            new_tree_node_visited = deepcopy(tree_node[1])
            if new_node in waypoints:
                new_tree_node_visited = list(new_tree_node_visited)
                new_tree_node_visited[waypoints.index(new_node)] = True
                new_tree_node_visited = tuple(new_tree_node_visited)
            new_tree_node = (new_node, new_tree_node_visited)

            # this condition is quite restrictive as it combines multiple paths into 1 and hence not desired for information gathering
            if new_tree_node in tree.nodes():
                continue
                
            new_gval = gval + get_cost(node, h, new_node)
            # this can be multiplied by an allowance factor to allow paths greater than shortest length
            if new_gval >= shortest_length:
                continue
            open_list.append(new_tree_node)
            gvals.append(new_gval)
            new_heading = get_heading(new_node, node)
            headings.append(new_heading)

            # add new node to tree
            tree.add_node(new_tree_node, pos=new_tree_node[0])
            tree.add_edge(tree_node, new_tree_node)

            if sum(new_tree_node_visited) == nw:
                shortest_length = min(new_gval, shortest_length)                

    print(shortest_length)
    terminals = [node for node in tree.nodes() if sum(node[1])==nw]
    all_paths = [nx.all_simple_paths(tree, tree_start_node, t) for t in terminals]




if __name__ == '__main__':  
    env = FieldEnv(data_file=None, num_rows=15, num_cols=15)
    # env.map.render()

    rows = env.map.row_pass_indices
    cols = np.arange(0, env.map.shape[1], 2)
    g = nx.Graph()
    # add nodes
    for r in rows:
        for c in cols:
            g.add_node((r,c), pos=(r,c))

    delta_x = env.map.stack_len + 1
    # x-axis is row or the first element of the tuple
    dxdy = [(delta_x,0), (-delta_x,0), (0,2), (0,-2)]

    # add edge
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
    waypoints = [(6,28), (10,16), (18,8)]
    heading = (1,0)
    get_shortest_path_waypoints(start, heading, waypoints, g, env.map)  
        