import numpy as np 
from utils import generate_gaussian_data, manhattan_distance, valid_neighbors, entropy_from_cov, compute_rmse
import ipdb
from models import GpytorchGPR
from networkx import nx
from graph_utils import edge_cost, get_heading, find_merge_to_node
import matplotlib.pyplot as plt
import torch

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)

def find_all_paths(sz, start, heading, goal, budget):
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


def find_best_path(all_paths, sz, cov_matrix, ind, entropy_const=None):
    all_ents = []
    for i in range(len(all_paths)):
        x = list(set(all_paths[i]))
        poses = np.vstack(x)        
        a = ind[poses[:,0], poses[:,1]]
        cov = cov_matrix[a].T[a].T
        ent = entropy_from_cov(cov, entropy_const)
        all_ents.append(ent)
    best = np.argmax(all_ents)
    return best


def find_best_path_mi(all_paths, sz, cov_matrix, ind):
    all_uts = []
    all_ind = set(range(sz**2))
    for i in range(len(all_paths)):
        x = list(set(all_paths[i]))
        poses = np.vstack(x)        
        a = ind[poses[:,0], poses[:,1]]
        a_bar = list(all_ind - set(a))
        cov = cov_matrix[a].T[a].T
        cov_abar = cov_matrix[a_bar].T[a_bar].T
        ut = entropy_from_cov(cov) + entropy_from_cov(cov_abar)
        all_uts.append(ut)
    best = np.argmax(all_uts)
    return best


def find_best_path_arv(all_paths, sz, cov_matrix, ind):
    all_uts = []
    for i in range(len(all_paths)):
        x = list(set(all_paths[i]))
        poses = np.vstack(x)        
        a = ind[poses[:,0], poses[:,1]]
        cov = cov_matrix[a].T[a].T
        cov_inv = np.linalg.inv(cov)
        cov_xa = cov_matrix[:,a]
        mat = np.dot(cov_xa, np.dot(cov_inv, cov_xa.T))
        ut = np.trace(mat)/(sz**2)
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


def draw_path(ax, path):
    head_width = .05
    head_length = .1
    linewidth = 2
    delta = head_length*2
    arrow_color = 'red'

    for i in range(len(path)-1):
        source = path[i]
        sink = path[i+1]
        dxdy = (sink[0]-source[0], sink[1]-source[1])
        dx = dxdy[0]
        dy = dxdy[1]
        if dx == 0:
            sign = dy//abs(dy)
            dy = sign*(abs(dy)-delta)
        else:
            sign = dx//abs(dx)
            dx = sign*(abs(dx)-delta)
        
        ax.arrow(source[0], source[1], dx, dy,
                 head_width=head_width, head_length=head_length,
                 linewidth=linewidth, color=arrow_color, alpha=1)


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


def predict(sz, cov_matrix, path, ind, y_all):
    a = list(set([ind[p] for p in path]))
    x = np.arange(sz**2)
    cov_aa = cov_matrix[a].T[a].T
    cov_xa = cov_matrix[x].T[a].T
    cov_xx = cov_matrix[x].T[x].T
    cov_aa_inv = np.linalg.inv(cov_aa)

    mu = np.dot(cov_xa, np.dot(cov_aa_inv, y_all[a]))
    rmse = compute_rmse(mu, y)
    return rmse



# show that using entropy as information gain criterion, informative planning is not suitable
sz = 4
x,y = generate_gaussian_data(sz, sz, k=2, min_var=.5, max_var=10)
gp = GpytorchGPR(lr=.1, max_iterations=200)

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

cov_matrix = gp.cov_mat(x) + 1e-5*np.eye(len(x))
eig_vals = np.linalg.eigvalsh(cov_matrix)
min_eig = min(eig_vals)
entropy_constant = -.5 * np.log(min_eig)

ind = np.arange(sz**2).reshape(sz,sz)
best_idx = find_best_path(all_paths, sz, cov_matrix, ind)
best_mod_idx = find_best_path(all_paths, sz, cov_matrix, ind, entropy_constant)
best_mi_idx = find_best_path_mi(all_paths, sz, cov_matrix, ind)
# best_arv_idx = find_best_path_arv(all_paths, sz, cov_matrix, ind)
paths = [all_paths[best_idx], all_paths[best_mi_idx], all_paths[best_mod_idx]]
plot(sz, start, goal, paths)

rmse = predict(sz, cov_matrix, all_paths[best_idx], ind, y)
rmse_mod = predict(sz, cov_matrix, all_paths[best_mod_idx], ind, y)
rmse_mi = predict(sz, cov_matrix, all_paths[best_mi_idx], ind, y)
# rmse_arv = predict(sz, cov_matrix, all_paths[best_arv_idx], ind, y)

print('RMSE entropy: ', rmse)
print('RMSE mutual information: ', rmse_mi)
print('RMSE modified entropy (ours): ', rmse_mod)
