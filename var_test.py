# test different levels of variance and see the effect of lots of noisy observations

from arguments import get_args
from env import FieldEnv
from agent import Agent
from utils import compute_rmse, compute_mae
from plots import get_mean_and_std

import ipdb
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

# np.random.seed(0)
# torch.manual_seed(0)


def compare_strategies():
    args = get_args()
    title = {'plant_width': 'Plant Width', 'plant_count': 'Crop Density', 'plant_height': 'Plant Height'}
    strategies = ['shortest', 'max_ent', 'equi_sample']
    results = [[] for _ in range(len(strategies))]
    
    sims = args.num_sims
    update = args.update
    criterion = args.criterion
    num_runs = args.num_runs
    slack = args.slack
    kappa = 5
    agents = []
    for i in range(sims):
        env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
        ll_noise = True
        master = Agent(env, args, learn_likelihood_noise=ll_noise)

        for t in range(len(strategies)):
            agent = Agent(env, args, learn_likelihood_noise=ll_noise, parent_agent=master, mobile_std=kappa*args.static_std)
            res = agent.run_ipp(num_runs=num_runs, criterion=criterion, mobile_enabled=True, update=update, slack=slack, strategy=strategies[t])
            results.append(res['rmse'])
            agents.append(agent)

    results = [np.stack(x) for x in results]
    x = np.stack([np.arange(1,num_runs+1) for _ in range(sims)])
    dct = {'x': x.flatten()}
    for st, res in zip(strategies, results):
        dct[st] = res.flatten()
    df = pd.DataFrame.from_dict(dct)
    
    ci = 25
    fig, ax = plt.subplot(1,1)
    for st in strategies:
        ax = sns.lineplot(x='x', y=st, data=df, ax=ax, ci=ci, label=st)
    plt.xticks(np.arange(1,num_runs+1))
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.title(title[args.phenotype])
    plt.xlabel('Number of batches')
    plt.ylabel('Test RMSE')
    plt.legend()
    plt.show()
    # Save full size figure manually
    ipdb.set_trace()

    if not args.eval_only:
        # save plot
        # plt.savefig(os.path.join(args.save_dir, args.phenotype+'.png'))
        
        # save results dataframe
        df.to_pickle(os.path.join(args.save_dir, args.phenotype+'.pkl'))

        with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
            json.dump(vars(args), f, indent=True)
    ipdb.set_trace()

def max_ent_tests():
    args = get_args()
    title = {'plant_width': 'Plant Width', 'plant_count': 'Crop Density', 'plant_height': 'Plant Height'}

    sims = args.num_sims
    update = args.update
    criterion = args.criterion
    num_runs = args.num_runs
    slack = args.slack
    kappas = [1,2,5,10]
    results = [[] for _ in range(len(kappas))]

    for i in range(sims):
        env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
        ll_noise = True
        master = Agent(env, args, learn_likelihood_noise=ll_noise)

        agents = []
        for t in range(len(kappas)):
            agent = Agent(env, args, learn_likelihood_noise=ll_noise, parent_agent=master, mobile_std=kappas[t]*args.static_std)
            res = agent.run_ipp(num_runs=num_runs, criterion=criterion, mobile_enabled=True, update=update, slack=slack, strategy='max_ent')
            results[t].append(res['rmse'])
            agents.append(agent)

    results = [np.stack(x) for x in results]
    x = np.stack([np.arange(1,num_runs+1) for _ in range(sims)])
    dct = {'x': x.flatten()}
    for k,res in zip(kappas, results):
        dct[str(k)] = res.flatten()

    df = pd.DataFrame.from_dict(dct)
    means, stds = get_mean_and_std(df, kappas=kappas)
    
    ci = 25
    fig, ax = plt.subplots(1,1)
    for k in kappas:
        ax = sns.lineplot(x='x', y=str(k), data=df, ax=ax, ci=ci, label='k = ' + str(k))

    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.title(title[args.phenotype])
    plt.xlabel('Number of batches')
    plt.ylabel('Test RMSE')
    plt.xticks(np.arange(1,args.num_runs+1))
    plt.legend()
    plt.show()
    # Save figure manually
    ipdb.set_trace()
    
    if not args.eval_only:
        # save plot
        # fig.savefig(os.path.join(args.save_dir, args.phenotype+'.png'), dpi=600)
        
        # save results dataframe
        df.to_pickle(os.path.join(args.save_dir, args.phenotype+'.pkl'))

        with open(os.path.join(args.save_dir, "args.json"), 'w') as f:
            json.dump(vars(args), f, indent=True)
    ipdb.set_trace()
    

def oracle():
    args = get_args()
    sims = 10
    rmses = []
    maes = []
    for i in range(sims):
        env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)

        ll_noise = True
        master = Agent(env, args, learn_likelihood_noise=ll_noise)
        mu, cov = master.predict_test()
        rms = compute_rmse(env.test_Y, mu)
        mae = compute_mae(env.test_Y, mu)
        rmses.append(rms)
        maes.append(mae)
    mean, std = np.mean(rmses), np.std(rmses)
    ipdb.set_trace()

if __name__ == '__main__':
    # compare_strategies()
    max_ent_tests()
    # oracle()