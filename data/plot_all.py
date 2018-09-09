import pickle
import ipdb
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# NOTE: this script was used to remove enteries with high zscore statistics.
# all the files in female_gene_data has incorporated these modifications and should not be changed further


path = 'female_gene_data/'
phenotypes = ['plant_width', 'plant_count', 'plant_height', 'leaf_fill', 'grvi']
ph_dicts = {}
all_ph_vals = {}

for ph in phenotypes:
	filename = path + ph + '_mean.pkl'
	with open(filename, 'rb') as fn:
		data = pickle.load(fn)
	ph_dicts[ph] = data
	
	all_ph_vals[ph] = data['Y'][data['valid']]

female = data['X'][data['valid']][:, -4:]
labels = np.arange(4).reshape(1,-1).repeat(len(female), 0)
category = ((female*labels).sum(1)).astype(int)

all_ph_vals['category'] = category
df = pd.DataFrame.from_dict(all_ph_vals)
# df['X'] = data['X'][data['valid']].tolist()
# pd.to_pickle(df, path + 'all_mean.pkl')
# take only those rows which satisfy the zscore criteria
# zs = stats.zscore(df)
# threshold = 3
# mask = (np.abs(zs) < threshold).all(axis=1)
# df = df[mask]

# sns.pairplot(data=df)
# plt.show()

# set all rejected samples from pickle file as invalid
# indices = np.arange(len(data['X']))
# temp = indices[data['valid']]
# invalid_indices = temp[np.where(~mask)[0]]
# ipdb.set_trace()
# save_path = 'female_gene_data_processed/'
# for ph in ph_dicts:
# 	ph_dicts[ph]['valid'][invalid_indices] = False
# 	filename = save_path + ph + '_mean.pkl'
# 	with open(filename, 'wb') as fn:
# 		pickle.dump(ph_dicts[ph], fn)