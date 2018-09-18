import numpy as np 
from utils import generate_gaussian_data, manhattan_distance, valid_neighbors, entropy_from_cov, compute_rmse
from utils import posterior_distribution_from_cov, get_monotonic_entropy_constant, draw_path
import ipdb
from models import GPR
from networkx import nx
from graph_utils import edge_cost, get_heading, find_merge_to_node
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
    

from arguments import get_args
from env import FieldEnv
from agent import Agent


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


def find_best_path(all_paths, cov_matrix, pose_to_index_matrix, train_ind, criterion='entropy', sigma=0):
    all_uts = []
    all_ind = set(train_ind)
    if criterion=='entropy':
        ent_const = None
    elif criterion == 'monotonic_entropy':
        ent_const = get_monotonic_entropy_constant(cov_matrix[train_ind].T[train_ind].T + sigma * np.eye(len(train_ind)))

    for i in range(len(all_paths)):
        a = path_to_indices(all_paths[i], pose_to_index_matrix)
        cov = cov_matrix[a].T[a].T + sigma * np.eye(len(a))
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
            ut = np.trace(mat)
        else:
            raise NotImplementedError

        all_uts.append(ut)
    best = np.argmax(all_uts)
    return best


def draw_grid(ax, sz, pose_to_index_matrix):
    radius = .1
    linewidth = 1.5
    color = 'green'
    testing_color = 'blue'
    alpha = .75
    for i in range(sz):
        for j in range(sz):
            pose = (i,j)
            ngh = valid_neighbors(pose, (sz,sz))
            if pose_to_index_matrix[(i,j)] == -1:
                node_color = testing_color
            else:
                node_color = color

            node = plt.Circle(pose, radius, color=node_color, alpha=alpha)
            
            ax.add_artist(node)
            for child in ngh:
                x = [pose[0], child[0]]
                y = [pose[1], child[1]]
                edge = plt.Line2D(x,y, linewidth=linewidth, color=color, alpha=alpha)
                ax.add_artist(edge)


def plot(sz, start, goal, paths, pose_to_index_matrix, titles=None):
    fig, ax = plt.subplots(1,len(paths), figsize=(6,4))
    for i in range(len(paths)):
        iax, path= ax[i], paths[i]
        iax.get_xaxis().set_visible(False)
        iax.get_yaxis().set_visible(False)
        iax.set_xlim(-1, sz)
        iax.set_ylim(-1, sz)
        iax.set_aspect(1)
        draw_grid(iax, sz, pose_to_index_matrix)    
        draw_path(iax, path)
        if titles is not None:
            iax.set_title(titles[i])
        iax.text(start[0]-.65,start[1]-.35, 'start')
        iax.text(goal[0]-.65, goal[0]+.35, 'goal')
    plt.show()


def predict(cov_mat, x_all, y_all, path, pose_to_index_matrix, test_ind):
    train_ind = path_to_indices(path, pose_to_index_matrix)
    train_y = y_all[train_ind]
    mu = posterior_distribution_from_cov(cov_mat, train_ind, train_y, test_ind)
    rmse = compute_rmse(mu, y_all[test_ind])
    return rmse


def path_to_indices(path, pose_to_index_matrix):
    indices = [pose_to_index_matrix[p] for p in path]
    indices = list(set([x for x in indices if x!=-1]))
    return indices


def get_covariance_matrix(x1, x2, ls):
    cov = np.zeros((x1.shape[0], x2.shape[0]))
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            w_diff = (x1[i] - x2[j]) / ls
            cov[i,j] = np.exp(-np.dot(w_diff, w_diff.T)/2)
    return cov



def best_path(env, cov, const, gp_indices, ls):
    all_ents = []
    all_monoents = []
    for ind in gp_indices:
        if len(ind) == 0:
            ent = -np.inf
            mono_ent = -np.inf
        else:
            cov_a = cov[ind].T[ind].T
            ent = entropy_from_cov(cov_a)
            mono_ent = entropy_from_cov(cov_a, const)
        all_ents.append(ent)
        all_monoents.append(mono_ent)

    maxent_idx = np.argmax(all_ents)
    maxmonoent_idx = np.argmax(all_monoents)
    return maxent_idx, maxmonoent_idx


if __name__ == '__main__':
    args = get_args()
    env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
    agent_common = Agent(env, args, learn_likelihood_noise=False)
    params = [x.exp() for x in agent_common.gp.model.parameters()][0].detach().squeeze().cpu().numpy()
    var_n = agent_common.gp.likelihood.log_noise.exp().item()
    print(params)
    
    start = (0,0)
    waypoints = [(16,46)]
    assert env.map_pose_to_gp_index_matrix[waypoints[0]] is not None 
    heading = (1,0)
    least_cost = env.bnb_shortest_path(start, heading, waypoints)
    budget = least_cost + 20
    paths_checkpoints, paths_indices, paths_cost = env.get_shortest_path_waypoints(start, heading, waypoints, least_cost, budget)
    # ipdb.set_trace()

    ls = params
    cov = get_covariance_matrix(env.X, env.X, ls) + var_n * np.eye(len(env.X))
    const = get_monotonic_entropy_constant(cov)
    maxent, maxmonoent = best_path(env, cov, const, paths_indices, ls)
    maxent_ind = paths_indices[maxent]
    maxmonoent_ind = paths_indices[maxmonoent]

    agent_ent = Agent(env, args, learn_likelihood_noise=False, parent_agent=agent_common)
    agent_monoent = Agent(env, args, learn_likelihood_noise=False, parent_agent=agent_common)
    agent_ent._add_samples(indices=maxent_ind, source='sensor')
    agent_monoent._add_samples(indices=maxmonoent_ind, source='sensor')

    agent_ent.update_model()
    agent_monoent.update_model()

    mu_ent, cov = agent_ent.predict_test()
    mu_monoent, cov = agent_monoent.predict_test()

    # cov_xx = get_covariance_matrix(env.test_X, env.test_X, ls) + var_n * np.eye(len(env.test_X))
    
    # # entropy prediction 
    # cov_xa = get_covariance_matrix(env.test_X, env.X[maxent_ind], ls)
    # cov_aa = cov[maxent_ind].T[maxent_ind].T + var_n * np.eye(len(maxent_ind))
    # train_y = env.Y[maxent_ind]
    # train_y_mean = np.mean(train_y)
    # mat1 = np.dot(cov_xa, np.linalg.inv(cov_aa))
    # mu_ent = np.dot(mat1, (train_y-train_y_mean)) + train_y_mean
    
    # # monotonic entropy prediction
    # cov_xa = get_covariance_matrix(env.test_X, env.X[maxmonoent_ind], ls)
    # cov_aa = cov[maxmonoent_ind].T[maxmonoent_ind].T + var_n * np.eye(len(maxmonoent_ind))
    # train_y = env.Y[maxmonoent_ind]
    # train_y_mean = np.mean(train_y)
    # mat1 = np.dot(cov_xa, np.linalg.inv(cov_aa))
    # mu_monoent = np.dot(mat1, (train_y-train_y_mean)) + train_y_mean
    

    rmse_ent = compute_rmse(env.test_Y, mu_ent)
    rmse_monoent = compute_rmse(env.test_Y, mu_monoent)
    print('RMSE Ent:',rmse_ent)
    print('RMSE MonoEnt:',rmse_monoent)
    ipdb.set_trace()

    # show that using entropy as information gain criterion, informative planning is not suitable
    # sz = 5
    # x,y = generate_gaussian_data(sz, sz, k=5, min_var=.5, max_var=5)
    # y = y * 10
    
    # gp = GPR(lr=.1, max_iterations=200, learn_likelihood_noise=False)
    # gp.fit(x,y)
    # params = [i for i in gp.model.parameters()][0].detach().squeeze().cpu().numpy()
    # # ipdb.set_trace()

    # # take out a held out test set
    # n = len(x)
    # num_test = n//5
    # perm = np.random.permutation(n)
    # test_ind = perm[:num_test]
    # train_ind = perm[num_test:]

    # # create a position to training index conversion matrix
    # conv = np.arange(n)
    # conv[test_ind] = -1
    # pose_to_index_matrix = conv.reshape(sz,sz)

    # ls = np.array([1,200])
    # cov_matrix = get_covariance_matrix(x,x,ls)
    
    # start = (0,0)
    # goal = (sz-1,sz-1)
    # budget = manhattan_distance(start, goal) + 5
    # heading = (1,0)
    # all_paths, all_paths_cost = find_all_paths((sz,sz), start, heading, goal, budget)
    # print('Number of paths:', len(all_paths))

    # ipdb.set_trace()

    # max_ent_idx = find_best_path(all_paths, cov_matrix, pose_to_index_matrix, train_ind, criterion='entropy', sigma=.01)
    # # max_mi_idx = find_best_path(all_paths, cov_matrix, pose_to_index_matrix, train_ind, criterion='mutual_information', sigma=.01)
    # max_monoent_idx = find_best_path(all_paths, cov_matrix, pose_to_index_matrix, train_ind, criterion='monotonic_entropy', sigma=.01)
    # # best_arv_idx = find_best_path(all_paths, sz, cov_matrix, ind, criterion='arv')

    
    # paths = [all_paths[max_ent_idx], all_paths[max_monoent_idx]]
    # # paths = [all_paths[max_ent_idx], all_paths[max_mi_idx], all_paths[best_arv_idx], all_paths[max_monoent_idx]]
    # titles = ['Entropy', 'Mutual Information', 'Monotonic Entropy']
    # plot(sz, start, goal, paths, pose_to_index_matrix, titles)

    # rmse_ent = predict(cov_matrix, x, y, all_paths[max_ent_idx], pose_to_index_matrix, test_ind)
    # # rmse_mi = predict(cov_matrix, x, y, all_paths[max_mi_idx], pose_to_index_matrix, test_ind)
    # rmse_monoent = predict(cov_matrix, x, y, all_paths[max_monoent_idx], pose_to_index_matrix, test_ind)
    # # rmse_arv = predict(x, y, ind, all_paths[best_arv_idx])

    # print('RMSE entropy: ', rmse_ent)
    # # print('RMSE mutual information: ', rmse_mi)
    # print('RMSE monotonic entropy: ', rmse_monoent)
    # # print('RMSE arv: ', rmse_arv)
    
    # ipdb.set_trace()