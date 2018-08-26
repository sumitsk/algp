from utils import load_data, zero_mean_unit_variance, normalize
from multi_task_models import MultitaskGP
from models import GpytorchGPR
import numpy as np
import gpflow
import pickle
from multi_spectralmixture import MultiSpectralMixture as MOSM

import ipdb

# this script carries out test to determine if multi-task learning is useful for the sorghum dataset or not
def test(x_train, y_train, x_test, y_test, model):
    num_tasks = len(y_train)
    
    if model == 'multi_task':
        gp = MultitaskGP(n_tasks=num_tasks, lr=.01, max_iter=1000)
        gp.fit(x_train, y_train)
        pred_mu, pred_rand_var = gp.predict(x_test)
        rmse = [np.linalg.norm(y_test[i] - pred_mu[i])/np.sqrt(len(y_test[i])) for i in range(num_tasks)]
        
    elif model == 'single':
        rmse = []
        for i in range(len(y_train)):
            gp = GpytorchGPR(lr=.01, max_iterations=1000)
            gp.fit(x_train, y_train[i])
            pred_y = gp.predict(x_test)
            rmse.append(np.linalg.norm(y_test[i] - pred_y) / np.sqrt(len(y_test[i])))

    elif model == 'spectral_mixture':
        number_of_components = 10
        kern = MOSM(x_train.shape[-1], num_tasks)
        for i in range(number_of_components-1):
            kern += MOSM(x_train.shape[-1], num_tasks)
        
        xtr = np.vstack([np.column_stack([np.full(shape=len(x_train), fill_value=i), x_train]) for i in range(num_tasks)])
        ytr = np.hstack(y_train)[:, None]
        xte = np.vstack([np.column_stack([np.full(shape=len(x_test), fill_value=i), x_test]) for i in range(num_tasks)])
        gp = gpflow.models.GPR(xtr, ytr, kern)        
        gpflow.train.ScipyOptimizer().minimize(gp, disp=False, maxiter=200)
        pred_y, pred_std = gp.predict_y(xte)  
        rmse = [np.linalg.norm(pred_y[xte[:,0]==i].squeeze() - y_test[i])/np.sqrt(len(y_test[i])) for i in range(num_tasks)]

    else:
        raise NotImplementedError

    return rmse


if __name__ == '__main__':
    # features = ['plant_width_mean', 'plant_height_mean', 'dry_to_green_ratio_mean', 'height_aerial']
    features = ['yield', 'dry_to_green_ratio_mean']
    
    results = []
    for f1 in features:
        for f2 in features:
            if f1 == f2:
                continue

            file1 = 'data/' + f1 + '_dataset.pkl'
            file2 = 'data/' + f2 + '_dataset.pkl'
            
            nr1, nc1, X1, Y1 = load_data(file1)
            nr2, nc2, X2, Y2 = load_data(file2)
            assert (nr1, nc1) == (nr2, nc2) and (X1==X2).all()
            X = X1

            # split into training and test set
            n = len(Y1)
            fraction = .6
            n_train = int(fraction*n)
            train_ind = np.random.randint(0, n, n_train)
            test_ind = list(set(list(range(n))) - set(train_ind))

            x_train = X[train_ind, :2]
            x_test = X[test_ind, :2]
            mean, std = x_train.mean(), x_train.std()
            x_train = zero_mean_unit_variance(x_train, mean, std)
            x_test = zero_mean_unit_variance(x_test, mean, std)
            
            y1_train = Y1[train_ind]
            y2_train = Y2[train_ind]
            y1_test = Y1[test_ind]
            y2_test = Y2[test_ind]
            
            # normalize y 
            # y1_max = y1_train.max()
            # y2_max = y2_train.max()
            # y1_train = normalize(y1_train, y1_max)
            # y2_train = normalize(y2_train, y2_max)
            # y1_test = normalize(y1_test, y1_max)
            # y2_test = normalize(y2_test, y2_max)

            y_train = [y1_train, y2_train]
            y_test = [y1_test, y2_test]
            
            rmse_single = test(x_train, y_train, x_test, y_test, 'single')
            rmse_multi = test(x_train, y_train, x_test, y_test, 'multi_task')
            rmse_sm = test(x_train, y_train, x_test, y_test, 'spectral_mixture')
            
            print('\n===============================================')
            print(f1, f2)
            print('Test rmse:')
            print('single', rmse_single)
            print('multi', rmse_multi)
            print('sm', rmse_sm)
            print('===============================================')
            
            res = dict(f1=f1, f2=f2, rmse_single=rmse_single, rmse_multi=rmse_multi, rmse_sm=rmse_sm)
            results.append(res)

    ipdb.set_trace()
    with open('multi_test_results.pkl', 'wb') as fn:
        pickle.dump(results, fn)
    
