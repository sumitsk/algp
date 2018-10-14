import numpy as np 
import pandas as pd
import seaborn as sns
from pprint import pprint

from env import FieldEnv
from agent import Agent
from arguments import get_args
from utils import compute_metric, normal_dist_kldiv, generate_lineplots
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
    
    results_maxent = []
    results_naive_static = []
    results_naive_mobile = []
    results_shortest = []
    results_equi_sample =[]

    nsims = 10
    k = 10
    num_naive_runs = 20
    disp = False
    for i in range(nsims):
        env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
        master = Agent(env, args)
        # master.pilot_survey(num_samples=k, std=master.static_std)
        
        # It is not necessary to make separate agents but is useful for debugging purposes
        agent_maxent = Agent(env, args, parent_agent=master)
        agent_shortest = Agent(env, args, parent_agent=master)
        agent_equi_sample = Agent(env, args, parent_agent=master)
        agent_naive_static = Agent(env, args, parent_agent=master)
        agent_naive_mobile = Agent(env, args, parent_agent=master)

        res_maxent = agent_maxent.run_ipp(num_runs=args.num_runs, strategy='MaxEnt', disp=disp) 
        res_shortest = agent_shortest.run_ipp(num_runs=args.num_runs, strategy='Shortest', disp=disp)
        res_equi_sample = agent_equi_sample.run_ipp(num_runs=args.num_runs, strategy='Equi-sample', disp=disp)
        res_naive_static = agent_naive_static.run_naive(std=master.static_std, counts=[k]*num_naive_runs, metric='distance')
        res_naive_mobile = agent_naive_mobile.run_naive(std=master.mobile_std, counts=[k]*num_naive_runs, metric='distance')
        
        # RMSE v/s distance travelled
        rmse_maxent, mi_maxent = agent_maxent.prediction_vs_distance(k=k, num_runs=num_naive_runs)
        rmse_shortest, mi_shortest = agent_shortest.prediction_vs_distance(k=k, num_runs=num_naive_runs)
        rmse_equi_sample, mi_equi_sample = agent_equi_sample.prediction_vs_distance(k=k, num_runs=num_naive_runs)

        results_maxent.append(rmse_maxent)
        results_shortest.append(rmse_shortest)
        results_equi_sample.append(rmse_equi_sample)
        results_naive_static.append(res_naive_static['rmse'])
        results_naive_mobile.append(res_naive_mobile['rmse'])

        ipdb.set_trace()

    # TODO: make all curves start at the same test RMSE
    # TODO: apart from test RMSE, also use MI (both remaining training set and test set) for comparison
    # NOTE: in some cases, there is significant difference between MaxEnt and two naive baselines

    start = k
    x = np.arange(start, start+k*num_naive_runs, k)
    x = np.stack([x for _ in range(nsims)]).flatten()
    ys = [np.stack(results_maxent).flatten(), np.stack(results_shortest).flatten(), np.stack(results_equi_sample).flatten(),
          np.stack(results_naive_static).flatten(), np.stack(results_naive_mobile).flatten()]
    xlabel = 'Distance travelled'
    ylabel = 'Test RMSE'
    legends = ['MaxEnt', 'Shortest', 'Equi-Sample', 'Naive Static', 'Naive Mobile']
    ci = 50
    generate_lineplots(x, ys, xlabel=xlabel, ylabel=ylabel, legends=legends, ci=ci)






    # Save arguments as json file
    # if not args.eval_only:
    #     with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
    #         json.dump(vars(args), f, indent=True)
