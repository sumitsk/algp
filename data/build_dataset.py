import pandas as pd
import ipdb
import pickle
import numpy as np

# mapping_df has Ranges in this order: 1,1,1.....,2,2...,37,37..
# converting it to 1,2,...37,1,2,...,37,1,2.....
mapping_df = pd.read_pickle('mapping.pkl')
males = mapping_df[['Male']].values.squeeze()
females = mapping_df[['Female']].values.squeeze()
males = males.reshape((37,15)).T.flatten()
females = females.reshape((37,15)).T.flatten()
mapping_ranges = mapping_df[['Range']].values.squeeze().reshape((37,15)).T.flatten()
mapping_row1 = mapping_df[['Row1']].values.squeeze().reshape((37,15)).T.flatten()


# male and female genotype category
with open('male_genotype_category.pkl', 'rb') as fn:
	male_genotype_category = pickle.load(fn)

female_genotype_category = {}
label = 0
for f in np.unique(females[females!=None]):
	if f == '83P17':
		continue
	female_genotype_category[f] = label
	label += 1


# converting categories to one-hot form
num_males = 4
num_females = 4
male_rep = np.zeros((len(males), num_males))
female_rep = np.zeros((len(females), num_females))
for i in range(len(males)):
	if males[i] is None:
		continue
	label = male_genotype_category[males[i]]
	male_rep[i, label] = 1

for i in range(len(females)):
	if females[i] is None or females[i]=='83P17':
		continue
	label = female_genotype_category[females[i]]
	female_rep[i, label] = 1

load_file = ['dry_to_green_ratio', 'grvi', 'height_ground_robot', 'leaf_fill', 'plant_count', 'plant_width']
load_file = ['raw_data/' + x +'_measures.pkl' for x in load_file]
save_file = ['dry_to_green_ratio', 'grvi', 'plant_height', 'leaf_fill', 'plant_count', 'plant_width']
save_file = ['female_gene_data/' + x + '_mean.pkl' for x in save_file]

feature = ['Mean', 'Mean', 'Avg_height_ground(in cm)', 'Mean', 'Mean', 'Mean']
div = [1, 1, 100, 1, 1, 1]


# ==============================================================================================================================
# save data by female genotype
for i in range(len(load_file)):
	# load raw data
	df = pd.read_pickle(load_file[i])
	df.columns = [x.strip() for x in df.columns]
	df_ranges = df[['Range']].values.squeeze()
	df_row1 = df[['Row1']].values.squeeze()
	vals = df[[feature[i]]].values.squeeze()/div[i]
	if not(np.equal(mapping_ranges, df_ranges).all() and np.equal(mapping_row1, df_row1).all()):
		df_ranges = df_ranges.reshape(37,15).T.flatten()
		df_row1 = df_row1.reshape(37,15).T.flatten()
		vals = vals.reshape(37,15).T.flatten()

	assert np.equal(mapping_ranges, df_ranges).all() and np.equal(mapping_row1, df_row1).all()	

	# in this data file, add an extra field valid which is False for all those id_abbreviated which we don't care about (FILL and '83P17')
	coords = np.vstack([df_row1, df_ranges]).T
	# x = np.concatenate([coords, male_rep, female_rep], axis=1)
	x = np.concatenate([coords, female_rep], axis=1)
	y = vals
	# atleast one of the parents is not None
	valid = x[:,2:].sum(1)>0

	data_dict = {'X': x, 'Y':y, 'valid': valid, 'num_rows': 15, 'num_cols': 37}

	with open(save_file[i], 'wb') as fn:
		pickle.dump(data_dict, fn)	

# ===================================================================================================================================
