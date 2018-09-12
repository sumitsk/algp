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
from methods import ground_truth
from copy import deepcopy

# seed = 0
# import random 
# random.seed(seed)
# import numpy as np
# np.random.seed(seed)
# import torch
# torch.manual_seed(seed)


def evaluate(agent, args, criterion, camera_enabled, adaptive, metric, true_values, oracle):
    results = agent.run_ipp(num_runs=args.num_runs, criterion=criterion, camera_enabled=camera_enabled, adaptive=adaptive)
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

if __name__ == '__main__':
    args = get_args()
    args_dict = vars(args)
    pprint(args_dict)
    
    # Save arguments as json file
    if not args.eval_only:
        with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
            json.dump(vars(args), f, indent=True)

    env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
    
    # get_heatmap(env)
    agent_common = Agent(env, args, learn_likelihood_noise=True)
    all_res = []
    all_res_camera = []

    # res1 = agent1.run_ipp(num_runs=args.num_runs, criterion='entropy', camera_enabled=True, adaptive=False, render=False)
    for _ in range(10):
        env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
        agent1 = Agent(env, args, parent_agent=agent_common, learn_likelihood_noise=True)
        res1 = agent1.run_ipp(num_runs=args.num_runs, criterion='monotonic_entropy', camera_enabled=False, adaptive=False, render=False)
        all_res.append([x['rmse'] for x in res1])
        
        agent2 = Agent(env, args, parent_agent=agent_common, learn_likelihood_noise=True)
        res2 = agent2.run_ipp(num_runs=args.num_runs, criterion='monotonic_entropy', camera_enabled=True, adaptive=False, render=False)
        all_res_camera.append([x['rmse'] for x in res2])

    ipdb.set_trace()

    # res3 = agent3.run_ipp(num_runs=args.num_runs, criterion='entropy', camera_enabled=False, adaptive=False, render=False)
    
    # oracle = ground_truth(env, args)
    # # only sensor, entropy
    # ent_rmse, ent_kldiv = evaluate(agent, args, criterion='entropy', camera_enabled=False, adaptive=False, metric='rmse', true_values=env.test_Y, oracle=oracle)

    # # only sensor, mutual information
    # # mi_rmse, mi_kldiv = evaluate(agent, args, criterion='mutual_information', camera_enabled=False, adaptive=False, metric='rmse', true_values=env.test_Y, oracle=oracle)

    # # both, monotonic entropy
    # both_mono_ent_rmse, both_mono_ent_kldiv = evaluate(agent, args, criterion='monotonic_entropy', camera_enabled=True, adaptive=False, metric='rmse', true_values=env.test_Y, oracle=oracle)
    

    # # both, entropy
    # agent2 = Agent(env, args)
    # both_ent_rmse, both_ent_kldiv = evaluate(agent2, args, criterion='entropy', camera_enabled=True, adaptive=False, metric='rmse', true_values=env.test_Y, oracle=oracle)
    
    
    # # both, mutual information 
    # # both_mi_rmse, both_mi_kldiv = evaluate(agent, args, criterion='mutual_information', camera_enabled=True, adaptive=False, metric='rmse', true_values=env.test_Y, oracle=oracle)
    

    # ipdb.set_trace()

    # plt.figure(0)
    # x = np.arange(1,args.num_runs+1)*args.num_samples_per_batch
    # plt.plot(x, ent_rmse, label='entropy')
    # # plt.plot(x, mi_rmse, label='mutual_information')
    # plt.plot(x, both_ent_rmse, label='both_entropy')
    # # plt.plot(x, both_mi_rmse, label='both_mutual_information')
    # plt.plot(x, both_mono_ent_rmse, label='both_monotonic_entropy')
    # plt.plot(x, np.full(len(x), oracle['rmse']), linestyle='--', label='oracle')
    # plt.legend()
    # plt.show()

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
    