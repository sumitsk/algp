# test different levels of variance and see the effect of lots of noisy observations

from arguments import get_args
from env import FieldEnv
from agent import Agent
import ipdb
from utils import compute_rmse, compute_mae
from methods import baseline
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plots import get_mean_and_std
# np.random.seed(0)
# torch.manual_seed(0)


def compare_strategies():
	args = get_args()

	results1 = []
	results2 = []
	results3 = []
	
	update=False
	criterion='entropy'
	num_runs = 8
	slack = 0

	sims = 5
	kappa = 5
	for i in range(sims):
		env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)

		ll_noise = True
		master = Agent(env, args, learn_likelihood_noise=ll_noise)

		agent1 = Agent(env, args, learn_likelihood_noise=ll_noise, parent_agent=master, camera_std=kappa*args.sensor_std)
		agent2 = Agent(env, args, learn_likelihood_noise=ll_noise, parent_agent=master, camera_std=kappa*args.sensor_std)
		agent3 = Agent(env, args, learn_likelihood_noise=ll_noise, parent_agent=master, camera_std=kappa*args.sensor_std)
	

		res1 = agent1.run_ipp(num_runs=num_runs, criterion=criterion, camera_enabled=True, update=update, slack=slack, strategy='shortest')
		res2 = agent2.run_ipp(num_runs=num_runs, criterion=criterion, camera_enabled=True, update=update, slack=slack, strategy='max_ent')
		res3 = agent3.run_ipp(num_runs=num_runs, criterion=criterion, camera_enabled=True, update=update, slack=slack, strategy='equi_sample')
	
		results1.append(res1['rmse'])
		results2.append(res2['rmse'])
		results3.append(res3['rmse'])
		
	results1 = np.stack(results1)
	results2 = np.stack(results2)
	results3 = np.stack(results3)
	
	x = np.stack([np.arange(1,num_runs+1) for _ in range(sims)])
	dct = {'x': x.flatten(), 'shortest': results1.flatten(), 'max_ent': results2.flatten(), 'equi_sample': results3.flatten()}
	df = pd.DataFrame.from_dict(dct)
	
	ci = 50
	ax = sns.lineplot(x='x', y='shortest', data=df, ci=ci, label='shortest')
	ax = sns.lineplot(x='x', y='max_ent', data=df, ax=ax, ci=ci, label='max_ent')
	ax = sns.lineplot(x='x', y='equi_sample', data=df, ax=ax, ci=ci, label='equi_sample')
	ipdb.set_trace()
	plt.xlabel('Number of batches')
	plt.ylabel('Test RMSE')
	plt.show()


def max_ent_tests():
	args = get_args()

	sims = 10

	results1 = []
	results2 = []
	results3 = []
	results4 = []
	results5 = []

	update=False
	criterion='entropy'
	num_runs = 10
	slack = 0

	for i in range(sims):
		env = FieldEnv(data_file=args.data_file, phenotype=args.phenotype, num_test=args.num_test)

		ll_noise = True
		master = Agent(env, args, learn_likelihood_noise=ll_noise)

		agent1 = Agent(env, args, learn_likelihood_noise=ll_noise, parent_agent=master, camera_std=1*args.sensor_std)
		agent2 = Agent(env, args, learn_likelihood_noise=ll_noise, parent_agent=master, camera_std=2*args.sensor_std)
		agent3 = Agent(env, args, learn_likelihood_noise=ll_noise, parent_agent=master, camera_std=5*args.sensor_std)
		agent4 = Agent(env, args, learn_likelihood_noise=ll_noise, parent_agent=master, camera_std=10*args.sensor_std)
		agent5 = Agent(env, args, learn_likelihood_noise=ll_noise, parent_agent=master, camera_std=20*args.sensor_std)

		res1 = agent1.run_ipp(num_runs=num_runs, criterion=criterion, camera_enabled=True, update=update, slack=slack, strategy='max_ent')
		res2 = agent2.run_ipp(num_runs=num_runs, criterion=criterion, camera_enabled=True, update=update, slack=slack, strategy='max_ent')
		res3 = agent3.run_ipp(num_runs=num_runs, criterion=criterion, camera_enabled=True, update=update, slack=slack, strategy='max_ent')
		res4 = agent4.run_ipp(num_runs=num_runs, criterion=criterion, camera_enabled=True, update=update, slack=slack, strategy='max_ent')
		res5 = agent5.run_ipp(num_runs=num_runs, criterion=criterion, camera_enabled=True, update=update, slack=slack, strategy='max_ent')

		results1.append(res1['rmse'])
		results2.append(res2['rmse'])
		results3.append(res3['rmse'])
		results4.append(res4['rmse'])
		results5.append(res5['rmse'])

		# b_res = baseline(env, args)
		# rel_rmse1 = compute_rmse(b_res['mean'], res1['mean'])
		# rel_rmse2 = compute_rmse(b_res['mean'], res2['mean'])
		# rel_rmse3 = compute_rmse(b_res['mean'], res3['mean'])
		# rel_rmse4 = compute_rmse(b_res['mean'], res4['mean'])
		# mu1, cov1 = agent1.predict_test()
		# mu2, cov2 = agent2.predict_test()
		# mu3, cov3 = agent3.predict_test()
		# mu4, cov4 = agent4.predict_test()
		# kl_div1 = normal_dist_kldiv(b_res['mean'], b_res['covariance'], mu1, cov1)
		# kl_div2 = normal_dist_kldiv(b_res['mean'], b_res['covariance'], mu2, cov2)
		# kl_div3 = normal_dist_kldiv(b_res['mean'], b_res['covariance'], mu3, cov3)
		# kl_div4 = normal_dist_kldiv(b_res['mean'], b_res['covariance'], mu4, cov4)

	results1 = np.stack(results1)
	results2 = np.stack(results2)
	results3 = np.stack(results3)
	results4 = np.stack(results4)
	results5 = np.stack(results5)

	x = np.stack([np.arange(1,num_runs+1) for _ in range(sims)])
	# dct = {'x': x.flatten(), 'shortest': results1.flatten(), 'max_ent': results2.flatten(), 'equi_sample': results3.flatten()}
	dct = {'x': x.flatten(), '1': results1.flatten(), '2': results2.flatten(), '5': results3.flatten(), '10': results4.flatten(), '20': results5.flatten()}
	df = pd.DataFrame.from_dict(dct)
	means, stds = get_mean_and_std(df)
	ipdb.set_trace()
	ci = 75
	ax = sns.lineplot(x='x', y='1', data=df, ci=ci)
	ax = sns.lineplot(x='x', y='2', data=df, ax=ax, ci=ci)
	ax = sns.lineplot(x='x', y='5', data=df, ax=ax, ci=ci)
	ax = sns.lineplot(x='x', y='10', data=df, ax=ax, ci=ci)
	ax = sns.lineplot(x='x', y='20', data=df, ax=ax, ci=ci)
	ipdb.set_trace()
	plt.xlabel('Number of batches')
	plt.ylabel('Test RMSE')
	plt.show()


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
	# max_ent_tests()
	oracle()