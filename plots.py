import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import ipdb
plt.rcParams.update({'font.size': 22})

# return mean 
def get_mean_and_std(df, idx=10, kappas=None):
	ind = df.index[df['x'] == idx].tolist()
	temp = df.iloc[ind]
	kappas = ['1', '2', '5', '10', '20'] if kappas is None else [str(k) for k in kappas]
	means = []
	stds = []
	for k in kappas:
		vals = temp[k].values
		m, s = vals.mean(0), vals.std(0)
		means.append(m)
		stds.append(s)
	return means, stds


def compare_strategies():
	# direc_true = 'outputs/intel_compare_strategies/update_true/'
	direc_false = 'outputs/intel_compare_strategies/update_false/'
	fn1_10 = 'k1_s10.pkl'
	fn1_20 = 'k1_s20.pkl'
	fn2_10 = 'k2_s10.pkl'
	fn2_20 = 'k2_s20.pkl'
	fn5_10 = 'k5_s10.pkl'
	fn5_20 = 'k5_s20.pkl'
	fn10_10 = 'k10_s10.pkl'
	fn10_20 = 'k10_s20.pkl'
	
	# df1_10 = pd.read_pickle(direc_true + fn1_10)
	# df1_20 = pd.read_pickle(direc_true + fn1_20)
	# df2_10 = pd.read_pickle(direc_true + fn2_10)
	# df2_20 = pd.read_pickle(direc_true + fn2_20)
	# df5_10 = pd.read_pickle(direc_true + fn5_10)
	# df5_20 = pd.read_pickle(direc_true + fn5_20)
	# df10_10 = pd.read_pickle(direc_true + fn10_10)
	# df10_20 = pd.read_pickle(direc_true + fn10_20)

	df1_10 = pd.read_pickle(direc_false + fn1_10)
	df1_20 = pd.read_pickle(direc_false + fn1_20)
	df2_10 = pd.read_pickle(direc_false + fn2_10)
	df2_20 = pd.read_pickle(direc_false + fn2_20)
	df5_10 = pd.read_pickle(direc_false + fn5_10)
	df5_20 = pd.read_pickle(direc_false + fn5_20)
	df10_10 = pd.read_pickle(direc_false + fn10_10)
	df10_20 = pd.read_pickle(direc_false + fn10_20)

	kappas = ['max_ent', 'equi_sample', 'shortest']

	means1_10, stds1_10 = get_mean_and_std(df1_10, idx=8, kappas=kappas) 
	means1_20, stds1_20 = get_mean_and_std(df1_20, idx=8, kappas=kappas) 
	means2_10, stds2_10 = get_mean_and_std(df2_10, idx=8, kappas=kappas) 
	means2_20, stds2_20 = get_mean_and_std(df2_20, idx=8, kappas=kappas) 
	means5_10, stds5_10 = get_mean_and_std(df5_10, idx=8, kappas=kappas) 
	means5_20, stds5_20 = get_mean_and_std(df5_20, idx=8, kappas=kappas) 
	means10_10, stds10_10 = get_mean_and_std(df10_10, idx=8, kappas=kappas) 
	means10_20, stds10_20 = get_mean_and_std(df10_20, idx=8, kappas=kappas) 
	ipdb.set_trace()

if __name__ == '__main__':
	compare_strategies()
	# direc_false = 'outputs/intel_update_false/'
	# fn25 = 'update_false_pt_25_new.pkl'
	# fn50 = 'update_false_pt_50_new.pkl'
	# fn75 = 'update_false_pt_75_new.pkl'
	# fn100 = 'update_false_pt_100_new.pkl'

	# df25 = pd.read_pickle(direc_false + fn25)
	# df50 = pd.read_pickle(direc_false + fn50)
	# df75 = pd.read_pickle(direc_false + fn75)
	# df100 = pd.read_pickle(direc_false + fn100)

	# means25, stds25 = get_mean_and_std(df25)
	# means50, stds50 = get_mean_and_std(df50)
	# means75, stds75 = get_mean_and_std(df75)
	# means100, stds100 = get_mean_and_std(df100)

	# mean_oracle = .44

	# # direc_true = 'outputs/intel_update_true/'
	# # fn04 = 'update_true_pt_04.pkl'
	# # fn12 = 'update_true_pt_12.pkl'
	# # fn20 = 'update_true_pt_20.pkl'
	
	# # df04 = pd.read_pickle(direc_true + fn04)
	# # df12 = pd.read_pickle(direc_true + fn12)
	# # df20 = pd.read_pickle(direc_true + fn20)

	# # # means04, stds04 = get_mean_and_std(df04, idx=9)
	# # # means12, stds12 = get_mean_and_std(df12, idx=7)
	# # # means20, stds20 = get_mean_and_std(df20, idx=5)

	# # means04, stds04 = get_mean_and_std(df04)
	# # means12, stds12 = get_mean_and_std(df12)
	# # means20, stds20 = get_mean_and_std(df20)

	# ci = 50
	# df = df100
	# ax = sns.lineplot(x='x', y='1', data=df, ci=ci, label='k = 1')
	# ax = sns.lineplot(x='x', y='2', data=df, ci=ci, label='k = 2')
	# ax = sns.lineplot(x='x', y='5', data=df, ci=ci, label='k = 5')
	# ax = sns.lineplot(x='x', y='10', data=df, ci=ci, label='k = 10')
	# ax = sns.lineplot(x='x', y='20', data=df, ci=ci, label='k = 20')
	# n = 10
	# x = np.arange(1,n+1)
	# ax.plot(x, [mean_oracle]*n, '--', label='Oracle')
	# ax.set_ylim([.4, 2.5])
	
	# plt.ylabel('Test RMSE')
	# plt.xlabel('Number of Batches')
	# plt.title('Fixed Hyperparameters')
	# plt.xticks(np.arange(1,11))
	# plt.legend()
	# plt.show()


	# ipdb.set_trace()
