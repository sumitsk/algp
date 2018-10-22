import numpy as np 
import pandas as pd
from pprint import pprint

from env import FieldEnv
from agent import Agent
from arguments import get_args
from utils import generate_lineplots, compute_mae
# import ipdb

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


def snr_test(args):
    # compute signal-to-noise ratio as computed from the fitted GP model
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
    return all_rho


def compare_all_strategies(args):
    # compare all 5 strategies on the same environment 
    strategies = ['MaxEnt', 'Shortest', 'Equi-Sample', 'Naive Static', 'Naive Mobile']
    ipp_strategies = ['MaxEnt', 'Shortest', 'Equi-Sample']
    naive_strategies = ['Naive Static', 'Naive Mobile']
    num_strategies = len(strategies)

    nsims = 20
    k = 10
    num_naive_runs = 20
    disp = False
    # set some initial samples
    initial_samples = 5

    error_results = [[] for _ in range(num_strategies)]
    mi_results = [[] for _ in range(num_strategies)]
    var_results = [[] for _ in range(num_strategies)]
    noise_ratio = 5
    for t in range(nsims):
        env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
        master = Agent(env, args, static_std=args.static_std)
        master.reset()
        master.pilot_survey(num_samples=initial_samples, std=master.static_std)
        mu, cov, zero_mi = master.predict(x=env.test_X, return_cov=True, return_mi=True)
        zero_error = compute_mae(mu, env.test_Y)
        zero_mean_var = np.diag(cov).mean()

        # It is not necessary to make separate agents but is useful for debugging purposes
        agents = [Agent(env, args, parent_agent=master, static_std=args.static_std, mobile_std=noise_ratio*args.static_std) for _ in range(num_strategies)]
        
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

    start = k
    x = [initial_samples] + list(np.arange(start, start+k*num_naive_runs, k))
    x = np.stack([x for _ in range(nsims)]).flatten()
    xlabel = 'Distance travelled'
    ci = 50
    
    # test error
    errors = [np.stack(res).flatten() for res in error_results]
    dct_err = {'x': x}
    for y, lbl in zip(errors, strategies):
        dct_err[lbl] = y
    df_err = pd.DataFrame.from_dict(dct_err)

    ylabel = 'Test MAE'
    generate_lineplots(df_err, x='x', xlabel=xlabel, ylabel=ylabel, legends=strategies, ci=ci)
    # ipdb.set_trace()

    # There dataframes are not necessary to store 
    # test mean variance
    # dct_var = {'x': x}
    # varss = [np.stack(res).flatten() for res in var_results]
    # for y, lbl in zip(varss, strategies):
    #     dct_var[lbl] = y
    # df_var = pd.DataFrame.from_dict(dct_var)
    # ylabel_var = 'Test Mean Variance'
    # generate_lineplots(df_var, x='x', xlabel=xlabel, ylabel=ylabel_var, legends=strategies, ci=ci)

    # # mutual information
    # dct_mi = {'x': x}
    # mis = [np.stack(res).flatten() for res in mi_results]
    # for y, lbl in zip(mis, strategies):
    #     dct_mi[lbl] = y
    # df_mi = pd.DataFrame.from_dict(dct_mi)
    # ylabel_mi = 'Mutual Information'
    # generate_lineplots(df_mi, x='x', xlabel=xlabel, ylabel=ylabel_mi, legends=strategies, ci=ci)
    
    # params = dict(master.gp.model.named_parameters())


def compare_maxent(args):
    nsims = 10

    k = 10
    num_naive_runs = 25
    disp = False
    # set some initial samples
    initial_samples = 5
    # noise_ratios = [1,2,5,10]
    # variants = ['k = ' + str(n) for n in noise_ratios]

    slacks = [0, 5, 10]
    variants = ['slack = ' + str(s) for s in slacks]
    
    nv = len(variants)
    error_results = [[] for _ in range(nv)]
    mi_results = [[] for _ in range(nv)]
    var_results = [[] for _ in range(nv)]

    for t in range(nsims):
        env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
        master = Agent(env, args, static_std=args.static_std)
        master.reset()
        master.pilot_survey(num_samples=initial_samples, std=master.static_std)
        mu, cov, zero_mi = master.predict(x=env.test_X, return_cov=True, return_mi=True)
        zero_error = compute_mae(mu, env.test_Y)
        zero_mean_var = np.diag(cov).mean()

        # It is not necessary to make separate agents but is useful for debugging purposes
        # agents = [Agent(env, args, parent_agent=master, static_std=args.static_std, mobile_std=kappa*args.static_std) for kappa in noise_ratios]
        agents = [Agent(env, args, parent_agent=master, static_std=args.static_std, mobile_std=5*args.static_std) for _ in range(nv)]
        
        for i in range(nv):
            res = agents[i].run_ipp(num_runs=args.num_runs, strategy='MaxEnt', disp=disp, slack=slacks[i])
            res = agents[i].prediction_vs_distance(k=k, num_runs=num_naive_runs)
            error_results[i].append([zero_error] + res['error'])            
            mi_results[i].append([zero_mi] + res['mi'])
            var_results[i].append([zero_mean_var] + res['mean_var'])

    start = k
    x = [initial_samples] + list(np.arange(start, start+k*num_naive_runs, k))
    x = np.stack([x for _ in range(nsims)]).flatten()
    xlabel = 'Distance travelled'
    ci = 50
    
    # test error
    errors = [np.stack(res).flatten() for res in error_results]
    dct_err = {'x': x}
    for y, lbl in zip(errors, variants):
        dct_err[lbl] = y
    df_err = pd.DataFrame.from_dict(dct_err)

    ylabel = 'Test MAE'
    generate_lineplots(df_err, x='x', xlabel=xlabel, ylabel=ylabel, legends=variants, ci=ci)

    # test variance
    dct_var = {'x': x}
    varss = [np.stack(res).flatten() for res in var_results]
    for y, lbl in zip(varss, variants):
        dct_var[lbl] = y
    df_var = pd.DataFrame.from_dict(dct_var)
    ylabel_var = 'Test Mean Variance'
    generate_lineplots(df_var, x='x', xlabel=xlabel, ylabel=ylabel_var, legends=variants, ci=ci)
    # ipdb.set_trace()

def run_demo(args):
    env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)
    agent = Agent(env, args, static_std=args.static_std, mobile_std=10*args.static_std)
    # Reset the agent before execution
    agent.reset()
    
    # Informative strategies
    # ipp_strategies = ['MaxEnt', 'Shortest', 'Equi-Sample']
    # Naive strategies
    # naive_strategies = ['Naive Static', 'Naive Mobile']

    agent.run_ipp(render=args.render, num_runs=args.num_runs, strategy='MaxEnt')


if __name__ == '__main__':
    args = get_args()
    pprint(vars(args))
    run_demo(args)
    # compare_all_strategies(args)
    # compare_maxent(args)


