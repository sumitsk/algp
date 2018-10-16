import numpy as np 
import pandas as pd
import seaborn as sns
from pprint import pprint

from env import FieldEnv
from agent import Agent
from arguments import get_args
from utils import normal_dist_kldiv, generate_lineplots, compute_mae
# from methods import ground_truth
import ipdb

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# seed = 0
# import random 
# random.seed(seed)
# import numpy as np
# np.random.seed(seed)
# import torch
# torch.manual_seed(seed)


# def evaluate(agent, args, criterion, mobile_enabled, metric, true_values, oracle):
#     results = agent.run_ipp(num_runs=args.num_runs, criterion=criterion, mobile_enabled=mobile_enabled)
#     kldiv = normal_dist_kldiv(results[-1]['mean'], results[-1]['covariance'], oracle['mean'], oracle['covariance'])
#     means = [x['mean'] for x in results]
#     err = compute_metric(true_values, means, metric=metric)
#     return err, kldiv


# def get_heatmap(env):
#     plot = env.map_pose_to_gp_index_matrix
#     n = np.max(env.category)
#     for i in range(plot.shape[0]):
#         for j in range(plot.shape[1]):
#             if env.map.occupied[i,j]:
#                 plot[i,j] = n + 1
#             elif plot[i,j] is None:
#                 plot[i,j] = n + 2
#             else:
#                 plot[i,j] = env.category[plot[i,j]]
#     ipdb.set_trace()


# def static_vs_both(args):
#     # compare static and static+mobile adaptive sampling setting
#     # on a held out test set, plot rmse vs number of static samples

#     args_dict = vars(args)
#     pprint(args_dict)
    
#     extra_features = []
    
#     ll_noise = True
#     all_res = []
#     all_res_mobile = []

#     sims = 3
#     for _ in range(sims):
#         env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, extra_features=extra_features, num_test=args.num_test)
#         master = Agent(env, args, learn_likelihood_noise=ll_noise)
    
#         agent1 = Agent(env, args, parent_agent=master, learn_likelihood_noise=ll_noise)
#         res1 = agent1.run_ipp(num_runs=2*args.num_runs, criterion='entropy', mobile_enabled=False, render=False)
#         all_res.append([x['rmse'] for x in res1])

#         agent2 = Agent(env, args, parent_agent=master, learn_likelihood_noise=ll_noise)
#         res2 = agent2.run_ipp(num_runs=args.num_runs, criterion='entropy', mobile_enabled=True, render=False)
#         all_res_mobile.append([x['rmse'] for x in res2])

#     r1 = np.stack(all_res)
#     x1 = np.arange(1, 2*args.num_runs+1)*args.num_samples_per_batch
#     x1all = np.stack([x1 for _ in range(sims)]).flatten()

#     rc = np.stack(all_res_mobile)
#     xc = np.arange(1, args.num_runs+1)*args.num_samples_per_batch
#     xcall = np.stack([xc for _ in range(sims)]).flatten()
#     dict1 = {'Static samples': x1all, 'RMSE': r1.flatten()}
#     dictc = {'Static samples': xcall, 'RMSE': rc.flatten()}
#     df1 = pd.DataFrame.from_dict(dict1)
#     dfc = pd.DataFrame.from_dict(dictc)
    
#     ax = sns.lineplot(x='Static samples', y='RMSE', data=df1, label='Static')
#     sns.lineplot(x='Static samples', y='RMSE', data=dfc, label='Static + Mobile', ax=ax)
#     # set these according to phenotype
#     plt.title('Plant Height')
#     plt.xlabel('Static Samples')
#     plt.ylabel('Test RMSE')
#     plt.show()
#     ipdb.set_trace()


def snr_test(args):
    nsims = 5
    all_rho = []
    extra_features = []

    for i in range(nsims):
        env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, extra_features=extra_features, num_test=args.num_test)
        master = Agent(env, args)
        params = dict(master.gp.model.named_parameters())
        ss = np.exp(params['kernel_covar_module.log_outputscale'].item())
        sn = np.exp(params['likelihood.log_noise'].item())
        rho = ss**2/sn**2
        all_rho.append(rho)
    ipdb.set_trace()


if __name__ == '__main__':
    args = get_args()
    # snr_test(args)
    strategies = ['MaxEnt', 'Shortest', 'Equi-Sample', 'Naive Static', 'Naive Mobile']
    ipp_strategies = ['MaxEnt', 'Shortest', 'Equi-Sample']
    naive_strategies = ['Naive Static', 'Naive Mobile']
    num_strategies = len(strategies)

    nsims = 10
    k = 10
    num_naive_runs = 25
    disp = False
    initial_samples = 5

    error_results = [[] for _ in range(num_strategies)]
    mi_results = [[] for _ in range(num_strategies)]
    var_results = [[] for _ in range(num_strategies)]
    
    for t in range(nsims):
        env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
        master = Agent(env, args, static_std=args.static_std, mobile_std=10*args.static_std)
        master.reset()
        master.pilot_survey(num_samples=initial_samples, std=master.static_std)
        mu, cov, zero_mi = master.predict(x=env.test_X, return_cov=True, return_mi=True)
        zero_error = compute_mae(mu, env.test_Y)
        zero_mean_var = np.diag(cov).mean()

        # It is not necessary to make separate agents but is useful for debugging purposes
        agents = [Agent(env, args, parent_agent=master) for _ in range(num_strategies)]
        
        for i in range(num_strategies):
            if strategies[i] in ipp_strategies:
                res = agents[i].run_ipp(num_runs=args.num_runs, strategy=strategies[i], disp=disp)
                res = agents[i].prediction_vs_distance(k=k, num_runs=num_naive_runs)
            elif strategies[i] in naive_strategies:
                std = master.static_std if 'Static' in strategies[i] else master.mobile_std
                res = agents[i].run_naive(std=std, counts=[k]*num_naive_runs, metric='distance')
            else:
                raise NotImplementedError
            error_results[i].append([zero_error] + res['error'])
            mi_results[i].append([zero_mi] + res['mi'])
            var_results[i].append([zero_mean_var] + res['mean_var'])

    # TODO: apart from test RMSE, also use MI (both remaining training set and test set) for comparison

    start = k
    x = [initial_samples] + list(np.arange(start, start+k*num_naive_runs, k))
    x = np.stack([x for _ in range(nsims)]).flatten()
    ys = [np.stack(res).flatten() for res in error_results]

    # test error
    dct_err = {'x': x}
    for y, lbl in zip(ys, strategies):
        dct_err[lbl] = y
    df_err = pd.DataFrame.from_dict(dct_err)

    xlabel = 'Distance travelled'
    ylabel = 'Test MAE'
    ci = 50
    generate_lineplots(df_err, x='x', xlabel=xlabel, ylabel=ylabel, legends=strategies, ci=ci)
    
    # There dataframes are not necessary to store 
    # test mean variance
    dct_var = {'x': x}
    varss = [np.stack(res).flatten() for res in var_results]
    for y, lbl in zip(varss, strategies):
        dct_var[lbl] = y
    df_var = pd.DataFrame.from_dict(dct_var)
    ylabel_var = 'Test Mean Variance'
    generate_lineplots(df_var, x='x', xlabel=xlabel, ylabel=ylabel_var, legends=strategies, ci=ci)

    # mutual information
    dct_mi = {'x': x}
    mis = [np.stack(res).flatten() for res in mi_results]
    for y, lbl in zip(mis, strategies):
        dct_mi[lbl] = y
    df_mi = pd.DataFrame.from_dict(dct_mi)
    ylabel_mi = 'Mutual Information'
    generate_lineplots(df_mi, x='x', xlabel=xlabel, ylabel=ylabel_mi, legends=strategies, ci=ci)
    
    params = dict(master.gp.model.named_parameters())
    ipdb.set_trace()









    # Save arguments as json file
    # if not args.eval_only:
    #     with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
    #         json.dump(vars(args), f, indent=True)
