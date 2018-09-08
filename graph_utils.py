import numpy as np
import ipdb
from utils import manhattan_distance


def edge_cost(node, heading, next_node):
    # heading must be one from {(0,1), (0,-1), (1,0), (-1,0)}
    next_heading = get_heading(next_node, node)
    if opposite_headings(heading, next_heading):
        return np.inf
    return manhattan_distance(node, next_node)


def get_heading(node, previous_node):
    # return heading when robot moves from previous_node to node
    diff = (node[0]-previous_node[0], node[1]-previous_node[1])
    # return None if both nodes overlap
    if diff == (0,0):
        return None

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
        down_junc = map.get_down_junction(node)
        up_junc = map.get_up_junction(node)
        down_node, up_node = get_down_and_up_nodes(node, new_nodes, down_junc, up_junc)
        new_edges.append((down_node, node))
        new_edges.append((node, up_node))
        remove_edges.append((down_junc, up_junc))   
    return new_nodes, new_edges, remove_edges


def get_down_and_up_nodes(node, others, down_junction, up_junction):
    # find nodes just above and below the current node
    down_node, up_node = down_junction, up_junction
    for nd in others:
        if nd == node:
            continue
        if in_between(nd, down_node, node):
            down_node = nd
        if in_between(nd, node, up_node):
            up_node = nd
    return down_node, up_node


def in_between(node, down_node, up_node):
    # check if node lies between down and up
    if node[1] == down_node[1] and node[1] == up_node[1]:
        if down_node[0] > node[0] > up_node[0]:
            return True
    return False
    

def lower_bound_path_cost(pose, waypoints):
    # return minimum cost of traversal to all the unvisited waypoints

    # boundaries of the bounding box
    all_locs = np.vstack(waypoints + [pose])
    # TODO: this can be further tightened :
    # 1. if only one waypoint left (find the exact distance to go)
    # 2. if there are waypoints in columns other than the robot's, then use nearest junction for bounding box computation
    mins = np.min(all_locs, axis=0)
    maxs = np.max(all_locs, axis=0)
    t = np.array(pose)
    a = t - mins
    b = maxs - t
    dist = sum(a) + sum(b) + min(a[0],b[0]) + min(a[1],b[1])
    return dist


def opposite_headings(h0, h1):
    # headings should be one from {(0,1), (0,-1), (1,0), (1,-1)}
    temp = h0[0]*h1[0] + h0[1]*h1[1]
    if temp==-1:
        return True
    return False

# Not useful anymore
# def upper_bound(start, start_heading, waypoints, graph, extra_cost):
#     nw = len(waypoints)
#     visited = [False]*nw
#     total_cost = 0

#     node = start
#     heading = start_heading

#     while sum(visited) != nw:
#         all_min_costs = np.full(nw, np.inf)
#         final_headings = [None]*nw
#         # find shortest path from node to all the unvisited waypoints and greedily choose the nearest one
#         for i in range(nw):
#             if visited[i]:
#                 continue
#             # minimum cost of all paths from node to the i^{th} waypoint
#             path_gen = nx.all_shortest_paths(graph, node, waypoints[i])
#             paths_costs = []
#             paths_final_heading = []
#             for path in path_gen:
#                 initial_heading = get_heading(path[1], path[0])
#                 final_heading = get_heading(path[-1], path[-2])
#                 cost = path_cost(path)
#                 # if initial heading is opposite of the current heading, then add extra cost
#                 if opposite_headings(initial_heading, heading):
#                     first_edge_length = manhattan_distance(path[0], path[1])
#                     cost += extra_cost - first_edge_length

#                 paths_costs.append(cost)
#                 paths_final_heading.append(final_heading)    

#             idx = np.argmin(paths_costs)
#             final_headings[i] = paths_final_heading[idx] 
#             # min cost from node to waypoints[i]   
#             all_min_costs[i] = paths_costs[idx]

#         # select the nearest waypoint as the next node
#         idx = np.argmin(all_min_costs)
#         visited[idx] = True
#         node = waypoints[idx]
#         heading = final_headings[idx]
#         total_cost += all_min_costs[idx]
#     return total_cost


def path_cost(path):
    # return total path length 
    # consecutive nodes of the path are either along x-axis or y-axis so distance = manhattan distance
    cost = 0
    for i in range(len(path)-1):
        cost += manhattan_distance(path[i], path[i+1])
    return cost


def find_merge_to_node(tree, node):
    # all nodes in the graph with same attributes as node
    all_nodes = [n for n in tree.nodes() if tree.node[n]==node]
    if len(all_nodes) > 1:
        ipdb.set_trace()
    if len(all_nodes) > 0:
        return all_nodes[0]
    return None   