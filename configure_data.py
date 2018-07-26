import pickle
import numpy as np 
import ipdb
from utils import vec_to_one_hot_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_features_data(contents, features):
    if not isinstance(features, list):
        features = [features]
    all_vals = []
    for f in features:
        idx = contents['features'].index(f)
        values = contents['data'][:, idx]
        all_vals.append(values)

    return np.stack(all_vals).T


def get_dataset(filename=None, target_feature=None):
    if filename is None:
        filename = 'data/data_2018.pkl'
    if target_feature is None:
        target_feature = 'plant_width_mean'

    with open(filename, 'rb') as f:
        contents = pickle.load(f)

    # input data[row, range, variety]
    # r1 looks like [2,4,6,....,2,4,6,...]
    r1 = get_features_data(contents, ['row1'])
    r2 = get_features_data(contents, ['range'])
    geno = get_features_data(contents, ['variety']).astype(np.int)
    variety = vec_to_one_hot_matrix(geno.squeeze())
    mat = np.hstack([r1, r2, variety])
    num_rows = int(r1.max()/2)
    num_cols = int(r2.max())
    assert num_rows * num_cols == len(r1), 'Size mismatch while generating dataset!'

    # output data
    output = get_features_data(contents, target_feature)

    data_dict = {'num_rows': num_rows,
                 'num_cols': num_cols,
                 'X': mat,
                 'Y': output}
    return data_dict


def save_dataset(filename, target_feature):
    data_dict = get_dataset(filename, target_feature)
    save_file = 'data/' + target_feature + '_dataset.pkl'
    with open(save_file, 'wb') as fn:
        pickle.dump(data_dict, fn)

def plot_variety_distribution(fl):
    with open(fl, 'rb') as f:
        contents = pickle.load(f)
    varieties = get_features_data(contents, 'variety').squeeze().reshape(15,37)
    
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(varieties, cmap=plt.get_cmap('RdBu', 4))
    fig.colorbar(im, cax=cax, orientation='vertical', ticks=np.arange(4))
    plt.show()
    
if __name__ == '__main__':
    fl = 'data/data_2018.pkl'
    # available features
    # ['row1', 'row2', 'range', 'variety',
    # 'height_aerial(cm)', 'plant_height_mean(cm)',
    # 'plant_height_max(cm)', 'dry_to_green_ratio_mean',
    # 'dry_to_green_ratio_variance', 'plant_count_mean',
    # 'plant_count_variance', 'plant_width_mean',
    # 'plant_width_variance', 'grvi_mean', 'grvi_var',
    # 'leaf_fill_mean', 'leaf_fill_var']


    # save_dataset(fl, 'plant_width_mean')
    # save_dataset(fl, 'plant_height_mean(cm)')
    # save_dataset(fl, 'height_aerial(cm)')
    # save_dataset(fl, 'plant_count_mean')
    plot_variety_distribution(fl)