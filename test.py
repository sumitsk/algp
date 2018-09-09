import numpy as np 
from utils import generate_gaussian_data, manhattan_distance, valid_neighbors, entropy_from_cov, compute_rmse, posterior_distribution, get_monotonic_entropy_constant, draw_path
import ipdb
from models import GPR
from networkx import nx
from graph_utils import edge_cost, get_heading, find_merge_to_node
import matplotlib.pyplot as plt


def find_all_paths(sz, start, heading, goal, budget):
    # return all paths from start to goal within budget
    tree = nx.DiGraph()
    root = 0
    tree.add_node(root, pose=start, heading=heading, gval=0)
    open_list = [root]
    closed_list = []
    idx = 0

    while len(open_list) > 0:
        parent_idx = open_list.pop(0)
        parent = tree.node[parent_idx]
        
        children = valid_neighbors(parent['pose'], sz)
        for child in children:
            cost = edge_cost(parent['pose'], parent['heading'], child)
            if cost == np.inf:
                continue

            new_gval = parent['gval'] + cost
            new_heading = get_heading(child, parent['pose'])

            togo = manhattan_distance(child, goal)
            if new_gval + togo > budget:
                continue

            new_tree_node = dict(pose=child, heading=new_heading, gval=new_gval)

            merge_to = find_merge_to_node(tree, new_tree_node)
            if merge_to is not None:
                tree.add_edge(parent_idx, merge_to, weight=cost)
                continue
            
            # add new node to tree
            idx = idx + 1
            tree.add_node(idx, **new_tree_node)
            tree.add_edge(parent_idx, idx, weight=cost)

            if child == goal:
                closed_list.append(idx)
            else:
                open_list.append(idx)
    
    all_paths_gen = [nx.all_shortest_paths(tree, root, t, weight='weight') for t in closed_list]
    all_paths = []
    all_paths_cost = []
    for path_gen in all_paths_gen:
        for i, path in enumerate(path_gen):
            locs = [tree.node[p]['pose'] for p in path]
            path_cost = tree.node[path[-1]]['gval']
            all_paths.append(locs)
            all_paths_cost.append(path_cost)
    return all_paths, all_paths_cost


def find_best_path(all_paths, sz, cov_matrix, ind, criterion='entropy'):
    all_uts = []
    all_ind = set(range(sz**2))
    if criterion=='entropy':
        ent_const = None
    elif criterion == 'monotonic_entropy':
        ent_const = get_monotonic_entropy_constant(cov_matrix)

    for i in range(len(all_paths)):
        a = path_to_indices(all_paths[i], ind)
        cov = cov_matrix[a].T[a].T
        if criterion == 'entropy' or criterion=='monotonic_entropy':
            ut = entropy_from_cov(cov, ent_const)    
        elif criterion=='mutual_information':
            a_bar = list(all_ind - set(a))
            cov_abar = cov_matrix[a_bar].T[a_bar].T
            ut = entropy_from_cov(cov) + entropy_from_cov(cov_abar)
        elif criterion=='arv':
            cov_inv = np.linalg.inv(cov)
            cov_xa = cov_matrix[:,a]
            mat = np.dot(cov_xa, np.dot(cov_inv, cov_xa.T))
            ut = np.trace(mat)/(sz**2)
        else:
            raise NotImplementedError

        all_uts.append(ut)
    best = np.argmax(all_uts)
    return best


def draw_grid(ax, sz):
    radius = .1
    linewidth = 1.5
    color = 'green'
    alpha = .2
    for i in range(sz):
        for j in range(sz):
            pose = (i,j)
            ngh = valid_neighbors(pose, (sz,sz))
            node = plt.Circle(pose, radius, color=color, alpha=alpha)
            ax.add_artist(node)
            for child in ngh:
                x = [pose[0], child[0]]
                y = [pose[1], child[1]]
                edge = plt.Line2D(x,y, linewidth=linewidth, color=color, alpha=alpha)
                ax.add_artist(edge)


def plot(sz, start, goal, paths):
    fig, ax = plt.subplots(1,len(paths), figsize=(6,4))
    for iax, path in zip(ax,paths):
        iax.get_xaxis().set_visible(False)
        iax.get_yaxis().set_visible(False)
        iax.set_xlim(-1, sz)
        iax.set_ylim(-1, sz)
        iax.set_aspect(1)
        draw_grid(iax, sz)    
        draw_path(iax, path)
    plt.show()


def predict(x_all, y_all, ind, path):
    a = path_to_indices(path, ind)
    train_x = x_all[a]
    train_y = y_all[a]
    mu = posterior_distribution(gp, train_x, train_y, x_all)
    rmse = compute_rmse(mu, y)
    return rmse


def path_to_indices(path, ind):
    indices = list(set([ind[p] for p in path]))
    return indices


if __name__ == '__main__':
    # show that using entropy as information gain criterion, informative planning is not suitable
    sz = 4
    x,y = generate_gaussian_data(sz, sz, k=2, min_var=.5, max_var=10)
    gp = GPR(lr=.1, max_iterations=200)

    # learn gp hyperparameters
    n = len(x)
    num_train = int(n)
    train_ind = np.random.permutation(n)[:num_train]
    gp.fit(x[train_ind],y[train_ind])

    start = (0,0)
    goal = (sz-1,sz-1)
    budget = manhattan_distance(start, goal)*2
    heading = (1,0)
    all_paths, all_paths_cost = find_all_paths((sz,sz), start, heading, goal, budget)

    # add small values to diagonal
    cov_matrix = gp.cov_mat(x1=x, var=1e-5)

    ind = np.arange(sz**2).reshape(sz,sz)
    best_idx = find_best_path(all_paths, sz, cov_matrix, ind, criterion='entropy')
    best_mod_idx = find_best_path(all_paths, sz, cov_matrix, ind, criterion='monotonic_entropy')
    best_mi_idx = find_best_path(all_paths, sz, cov_matrix, ind, criterion='mutual_information')
    best_arv_idx = find_best_path(all_paths, sz, cov_matrix, ind, criterion='arv')

    paths = [all_paths[best_idx], all_paths[best_mi_idx], all_paths[best_mod_idx]]
    paths = [all_paths[best_idx], all_paths[best_mi_idx], all_paths[best_arv_idx], all_paths[best_mod_idx]]
    plot(sz, start, goal, paths)

    rmse = predict(x, y, ind, all_paths[best_idx])
    rmse_mod = predict(x, y, ind, all_paths[best_mod_idx])
    rmse_mi = predict(x, y, ind, all_paths[best_mi_idx])
    rmse_arv = predict(x, y, ind, all_paths[best_arv_idx])

    print('RMSE entropy: ', rmse)
    print('RMSE mutual information: ', rmse_mi)
    print('RMSE arv: ', rmse_arv)
    print('RMSE modified entropy (ours): ', rmse_mod)

    ipdb.set_trace()