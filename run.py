from env import FieldEnv
from agent import Agent
from arguments import get_args
from pprint import pprint
import os
import json
import numpy as np 

import matplotlib.pyplot as plt
import ipdb
from utils import compute_metric, normal_dist_kldiv
# from methods import ground_truth
from copy import deepcopy

# seed = 0
# import random 
# random.seed(seed)
# import numpy as np
# np.random.seed(seed)
# import torch
# torch.manual_seed(seed)

import pandas as pd
import seaborn as sns

plt.rcParams.update({'font.size': 22})
    


def evaluate(agent, args, criterion, mobile_enabled, metric, true_values, oracle):
    results = agent.run_ipp(num_runs=args.num_runs, criterion=criterion, mobile_enabled=mobile_enabled)
    kldiv = normal_dist_kldiv(results[-1]['mean'], results[-1]['covariance'], oracle['mean'], oracle['covariance'])
    means = [x['mean'] for x in results]
    err = compute_metric(true_values, means, metric=metric)
    return err, kldiv


def get_heatmap(env):
    plot = env.map_pose_to_gp_index_matrix
    n = np.max(env.category)
    for i in range(plot.shape[0]):
        for j in range(plot.shape[1]):
            if env.map.occupied[i,j]:
                plot[i,j] = n + 1
            elif plot[i,j] is None:
                plot[i,j] = n + 2
            else:
                plot[i,j] = env.category[plot[i,j]]
    ipdb.set_trace()


def static_vs_both(args):
    # compare static and static+mobile adaptive sampling setting
    # on a held out test set, plot rmse vs number of static samples

    args_dict = vars(args)
    pprint(args_dict)
    
    env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
    
    ll_noise = True
    agent_common = Agent(env, args, learn_likelihood_noise=ll_noise)
    all_res = []
    all_res_mobile = []

    sims = 3
    for _ in range(sims):
        env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
        agent1 = Agent(env, args, parent_agent=agent_common, learn_likelihood_noise=ll_noise)
        res1 = agent1.run_ipp(num_runs=2*args.num_runs, criterion='monotonic_entropy', mobile_enabled=False, render=False)
        all_res.append([x['rmse'] for x in res1])

        agent2 = Agent(env, args, parent_agent=agent_common, learn_likelihood_noise=ll_noise)
        res2 = agent2.run_ipp(num_runs=args.num_runs, criterion='monotonic_entropy', mobile_enabled=True, render=False)
        all_res_mobile.append([x['rmse'] for x in res2])

    r1 = np.stack(all_res)
    x1 = np.arange(1, 2*args.num_runs+1)*args.num_samples_per_batch
    x1all = np.stack([x1 for _ in range(sims)]).flatten()

    rc = np.stack(all_res_mobile)
    xc = np.arange(1, args.num_runs+1)*args.num_samples_per_batch
    xcall = np.stack([xc for _ in range(sims)]).flatten()
    dict1 = {'Static samples': x1all, 'RMSE': r1.flatten()}
    dictc = {'Static samples': xcall, 'RMSE': rc.flatten()}
    df1 = pd.DataFrame.from_dict(dict1)
    dfc = pd.DataFrame.from_dict(dictc)
    
    ax = sns.lineplot(x='Static samples', y='RMSE', data=df1, label='Static')
    sns.lineplot(x='Static samples', y='RMSE', data=dfc, label='Static + Mobile', ax=ax)
    # set these according to phenotype
    plt.title('Plant Height')
    plt.xlabel('Static Samples')
    plt.ylabel('Test RMSE')
    plt.show()
    ipdb.set_trace()


if __name__ == '__main__':
    args = get_args()

    results_ipp = []
    results_naive_static = []
    results_naive_mobile = []
    nsims = 1

    for i in range(nsims):
        env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
        agent_common = Agent(env, args)    

        agent1 = Agent(env, args, parent_agent=agent_common)
        agent2 = Agent(env, args, parent_agent=agent_common)
        agent3 = Agent(env, args, parent_agent=agent_common)

        r1 = agent1.run_ipp(mobile_enabled=True, render=True)
        ipdb.set_trace()
                

        r2 = agent2.run_naive(num_samples=r1['count'], source='static')
        r3 = agent2.run_naive(num_samples=r1['count'], source='mobile')

        results_ipp.append(r1['rmse'])
        results_naive_static.append(r2['rmse'])
        results_naive_mobile.append(r3['rmse'])

    
    # Save arguments as json file
    # if not args.eval_only:
    #     with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
    #         json.dump(vars(args), f, indent=True)

    # keys = [i for i in args_dict]
    # if not args.eval_only:
    #     # make a new workbook if not exists
    #     import datetime
    #     now = datetime.datetime.now()
    #     date = str(now.month)+'/'+str(now.day)
    #     time = str(now.hour)+':'+str(now.minute)
    #     args_dict['date'] = date
    #     args_dict['time'] = time
    #     try:
    #         rb = open_workbook(args.logs_wb)
    #     except Exception:
    #         import xlsxwriter
    #         wb = xlsxwriter.Workbook(args.logs_wb)
    #         sh = wb.add_worksheet()
    #         for i, val in enumerate(keys):
    #             sh.write(0, i, val)
    #         wb.close()
    #         rb = open_workbook(args.logs_wb)
        
    #     rsh = rb.sheets()[0]
    #     row = rsh.nrows
    #     wb = copy(rb)
    #     wsh = wb.get_sheet(0)
    #     for i in range(len(keys)):
    #         k = rsh.cell(0, i).value
    #         wsh.write(row, i, args_dict[k])    

    #     wb.save(args.logs_wb)    
    