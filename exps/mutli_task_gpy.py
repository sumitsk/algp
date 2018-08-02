from utils import load_data, zero_mean_unit_variance
from multi_task_models import MultitaskGP
from models import GpytorchGPR
import numpy as np
import ipdb


file1 = 'data/plant_width_mean_dataset.pkl'
# file2 = 'data/plant_count_mean_dataset.pkl'
# file2 = 'data/plant_height_mean(cm)_dataset.pkl'
# file2 = 'data/dry_to_green_ratio_mean_dataset.pkl'
file2 = 'data/height_aerial(cm)_dataset.pkl'


nr1, nc1, X1, Y1 = load_data(file1)
nr2, nc2, X2, Y2 = load_data(file2)
assert (nr1, nc1) == (nr2, nc2) and (X1==X2).all()

n = len(Y1)
n_train = int(.4*n)
train_ind = np.random.randint(0, n, n_train)
test_ind = list(set(list(range(n))) - set(train_ind))

x_train = X1[train_ind, :2]
x_test = X1[test_ind, :2]
mean = x_train.mean()
std = x_train.std()

x_train = zero_mean_unit_variance(x_train, mean, std)
x_test = zero_mean_unit_variance(x_test, mean, std)
# add variety component also
# x_train = np.hstack([x_train, X1[train_ind, 2:]])
# x_test = np.hstack([x_test, X1[test_ind, 2:]])
y1_train = Y1[train_ind]
y2_train = Y2[train_ind]
y1_test = Y1[test_ind]
y2_test = Y2[test_ind]
y1_max = y1_train.max()
y2_max = y2_train.max()
y1_train /= y1_max
y2_train /= y2_max
y1_test /= y1_max
y2_test /= y2_max

gp = MultitaskGP(n_tasks=2, lr=.01, max_iter=1000)
gp.fit(x_train, [y1_train, y2_train])
pred_mu, pred_rand_var = gp.predict(x_test)

rmse0 = np.linalg.norm(y1_test - pred_mu[0])/np.sqrt(len(y1_test))
rmse1 = np.linalg.norm(y2_test - pred_mu[1])/np.sqrt(len(y2_test))


gp1 = GpytorchGPR(lr=.01, max_iter=1000)
gp1.fit(x_train, y1_train)
pred_y1 = gp1.predict(x_test)
rmse11 = np.linalg.norm(y1_test - pred_y1)/np.sqrt(len(y1_test))

gp2 = GpytorchGPR(lr=.01, max_iter=1000)
gp2.fit(x_train, y2_train)
pred_y2 = gp2.predict(x_test)
rmse21 = np.linalg.norm(y2_test - pred_y2)/np.sqrt(len(y2_test))

ipdb.set_trace()