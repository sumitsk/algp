import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ipdb
import pickle


def get_parents(mapping_df, females_self_pollinating, females, hybrid):
    # return male and female of all the hyrids

    if hybrid == 'FILL' or hybrid == 'CHECK':
        return None, None
    
    # there are some genotypes of only female
    if hybrid in females_self_pollinating:
        return None, females[females_self_pollinating.index(hybrid)]
    
    # return male and female of the parent hybrid plant
    idx = mapping_df.index[mapping_df['ID_abbreviated']==hybrid].tolist()
    idx1 = mapping_df.index[mapping_df['ID_abbreviated']=='PI'+hybrid].tolist()
    if len(idx) == 0 and len(idx1)==0:
        return None, None
    elem = idx[0] if len(idx1)==0 else idx1[0]
    male = mapping_df.iloc[elem]['Male']
    female = mapping_df.iloc[elem]['Female']
    if male != male:
        male = None
    if female != female:
        female = None
    return male, female


def get_males_and_females():
    # return male, female and hybrid genotypes

    # mapping between abbreviated id, pedigree, male and female
    workbook = 'HDP_MapFL18.xlsx'
    sheet = 'ID_conversion'
    mapping_df = pd.read_excel(workbook, sheet_name=sheet)

    unknowns = ['83P17']
    females_geno = ['ATx623', 'ATx642', 'ATxARG-1', 'PHA86']
    females_self_pollinating = ['BTx623', 'BTx642', 'BTxARG-1', 'PHB86']

    data_file = 'raw_data/height_ground_robot_measures.pkl'
    f_df = pd.read_pickle(data_file)

    # make a dataframe with columns as Row1 Row2 Range ID_abbreviated Male Female
    hybrids = f_df[['ID_abbreviated']].values.squeeze()
    males = []
    females = []
    for h in hybrids:
        m, f = get_parents(mapping_df, females_self_pollinating + unknowns, females_geno + unknowns, h)
        males.append(m)
        females.append(f)
        
    males = np.array(males)
    females = np.array(females)
    return males, females, hybrids

    # males_none = np.where(males==None)
    # males_none_hybrids = np.unique(hybrids[males_none])
    # # males None correponds to these hybrids -> ['83P17', 'BTx623', 'BTx642', 'BTxARG-1', 'FILL', 'PHB86']
    # # among which everyone except '83P17' (don't know what is this) is female. 

    # all_males = np.unique(males[males != None])
    # count_males = [np.sum(males==m) for m in all_males]
    # # a small test to determine if the hybrid to male detection is correct or not
    # # for h, m in zip(hybrids, male):
    # #     if m is None and h not in females_self_pollinating + unknowns + ['FILL']:
    # #         print(h,m)

    # all_females = np.unique(females[females!=None])
    # count_females = [np.sum(females==f) for f in all_females]
    # print(all_males)
    # print(count_males)
    # print(all_females)
    # print(count_females)


def build_dataframe():
    # mapping between abbreviated id, pedigree, male and female
    males, females, hybrids = get_males_and_females()
    data_file = 'raw_data/height_ground_robot_measures.pkl'
    df = pd.read_pickle(data_file)
    new_df = df[['Row1', 'Row2', 'Range', 'ID_abbreviated']]
    new_df['Male'] = males
    new_df['Female'] = females
    new_df.to_pickle('mapping.pkl')
    

def get_stats(values, all_males, all_females):
    # return male and female dictionaries where each key (genotype) contains a list of phenotype measurements
    male_dct = {}
    female_dct = {}

    for m in np.unique(all_males[all_males!=None]):
        if m is not None:
            male_dct[m] = []
    
    for f in np.unique(all_females[all_females!=None]):
        if f is not None:
            female_dct[f] = []

    for i in range(len(values)):
        if all_males[i] is not None:
            male_dct[all_males[i]].append(values[i])
        if all_females[i] is not None:
            female_dct[all_females[i]].append(values[i])
    
    return male_dct, female_dct


def get_mean_and_std(male_dct):
    # return mean and std of all genotypes (keys of male_dct dictionary)

    # male genotypes bar graph
    means = np.array([np.mean(male_dct[k]) for k in male_dct])
    stds = np.array([np.std(male_dct[k]) for k in male_dct])
    male_genotypes = np.array(list(male_dct.keys()))

    # sort male genotypes alphabetically to maintain consistency 
    order = np.argsort(male_genotypes)
    male_genotypes = male_genotypes[order]
    means = means[order]
    stds = stds[order]
    return male_genotypes, means, stds


if __name__ == '__main__':
    males, females, hybrids = get_males_and_females()

    feature = 'plant_width'
    filename = 'raw_data/' + feature + '_measures.pkl'
    df = pd.read_pickle(filename)
    df.columns = [x.strip() for x in df.columns]
    col = 'Mean'
    pw_values = df[[col]].values.squeeze()
    male_dct, female_dct = get_stats(pw_values, males, females)
    
    # pw_genos, pw_means, pw_stds = get_mean_and_std(male_dct) 
    # dct = {'Plant width mean': pw_means, 
    #        'Plant width std': pw_stds,
    #        'Male genotype': pw_genos}

    # temp = pd.DataFrame.from_dict(dct)
    # sns.barplot(x='Male genotype', y='Plant width mean', data=temp)
    # plt.show()

    # female genotypes bar graph
    # female_dct.pop('83P17', None)
    # female_genotypes, means, stds = get_mean_and_std(female_dct)
    # dct = {'Plant width mean': means, 
    #        'Plant width std': stds,
    #        'Female genotype': female_genotypes}

    temp_dict = {'Females': females, feature: pw_values}
    temp_df = pd.DataFrame.from_dict(temp_dict)
    g = sns.catplot(x='Females', y=feature, data=temp_df)

    # temp_dict = {'Males': males, 'Plant width': pw_values}
    # temp_df = pd.DataFrame.from_dict(temp_dict)
    # g = sns.catplot(x='Males', y='Plant width', data=temp_df)


    # temp = pd.DataFrame.from_dict(dct)
    # sns.barplot(x='Female genotype', y='Plant width mean', data=temp, ax=g.ax)
    plt.show()

    # pw_indices = np.argsort(pw_means)
    # plt.scatter(np.arange(len(pw_means)), pw_means[pw_indices])
    # plt.show()

    # # assign each male genotype a category
    # pw_genos_sorted = pw_genos[pw_indices]
    # male_genotype_category = {}
    # for i in range(len(pw_genos)):
    #     male_genotype_category[pw_genos_sorted[i]] = min(i//10,3)
    # with open('male_genotype_category.pkl', 'wb') as fn:
    #     pickle.dump(male_genotype_category, fn)

    # filename = 'raw_data/plant_count_measures.pkl'
    # df = pd.read_pickle(filename)
    # df.columns = [x.strip() for x in df.columns]
    # pc_values = df[['Mean']].values.squeeze()

    # pc_genos, pc_means, pc_stds = get_mean_and_std(pc_values, males, females) 
    # pc_indices = np.argsort(pc_means)
    # plt.scatter(np.arange(len(pc_means)), pc_means[pc_indices])
    # plt.show()
