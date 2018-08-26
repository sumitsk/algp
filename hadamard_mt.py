import math
import torch
import gpytorch
from matplotlib import pyplot as plt

from gpytorch.kernels import RBFKernel, IndexKernel
from gpytorch.means import ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

import numpy as np
from utils import to_torch, generate_gaussian_data, zero_mean_unit_variance, load_data
import ipdb

class MultitaskGP(torch.nn.Module):
    def __init__(self, n_tasks=2):
        super(MultitaskGP, self).__init__()
        self.lr = .05
        self.n_tasks = n_tasks

    def reset(self, x, y_ind, y):
        self._train_x = to_torch(x)
        self._train_y = to_torch(y)
        self._train_y_ind = to_torch(y_ind).long()
        # self._train_y_mean = torch.FloatTensor([self._train_y[self._train_y_ind==i].mean().item() for i in range(self.n_tasks)])
        # self._norm_train_y = self._train_y - self._train_y_mean.gather(0, self._train_y_ind)
        self._train_y_mean = self._train_y.mean()
        self._norm_train_y = self._train_y - self._train_y_mean

        self.likelihood = GaussianLikelihood()
        self.model = MultitaskGPModel((self._train_x, self._train_y_ind), self._norm_train_y, self.likelihood, self.n_tasks)
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}, ], lr=self.lr)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)      

    def fit(self, x, y_ind, y):
        self.reset(x, y_ind, y)
        self.likelihood.train()
        self.model.train()

        n_iterations = 500
        for i in range(n_iterations):
            self.optimizer.zero_grad()
            output = self.model(self._train_x, self._train_y_ind)
            loss = -self.mll(output, self._norm_train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iterations, loss.data[0]))
            self.optimizer.step()

    def predict(self, x, y_ind):
        self.model.eval()
        self.likelihood.eval()
        x_ = to_torch(x)
        ind_ = to_torch(y_ind).long()

        with torch.no_grad():
            pred_y = self.likelihood(self.model(x_, ind_))
        # mean = (pred_y.mean().cpu() + self._train_y_mean.gather(0, ind_)).numpy()
        mean = (pred_y.mean().cpu() + self._train_y_mean).numpy()
        return mean


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = RBFKernel()
        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = IndexKernel(n_tasks=n_tasks, rank=1)

    def forward(self,x,i):
        mean_x = self.mean_module(x)
        # Get all covariances, we'll look up the task-specific ones
        covar_x = self.covar_module(x)
        # # Get the covariance for task i
        covar_i = self.task_covar_module(i)
        covar_xi = covar_x.mul(covar_i)
        return GaussianRandomVariable(mean_x, covar_xi)


if __name__ == '__main__':
    num_rows = num_cols = 20
    X1, Y1 = generate_gaussian_data(num_rows, num_cols)
    X2, Y2 = generate_gaussian_data(num_rows, num_cols)
    assert np.equal(X1,X2).all()
    
    # file1 = 'data/plant_width_mean_dataset.pkl'
    # file2 = 'data/height_aerial(cm)_dataset.pkl'

    # nr1, nc1, X1, Y1 = load_data(file1)
    # nr2, nc2, X2, Y2 = load_data(file2)
    # assert (nr1, nc1) == (nr2, nc2) and (X1==X2).all()
    # num_rows, num_cols = nr1, nc1
    X = X1

    # randomly select some training points
    n_train = 200
    n = len(X1)
    ind1 = np.random.randint(0, n, n_train)
    x1 = X[ind1, :]
    y1 = Y1[ind1]
    y1_max = y1.max()
    y1_ind = np.zeros(n_train)

    ind2 = np.random.randint(0, n, n_train)
    x2 = X[ind2, :]
    y2 = Y2[ind2]
    y2_max = y2.max()
    y2_ind = np.ones(n_train)

    train_x = np.concatenate([x1,x2])
    train_y = np.concatenate([y1/y1_max,y2/y2_max])
    train_y_ind = np.concatenate([y1_ind, y2_ind])

    gp = MultitaskGP(n_tasks=2)
    gp.fit(train_x, train_y_ind, train_y)
    pred1 = gp.predict(X, np.zeros_like(Y1))
    pred2 = gp.predict(X, np.ones_like(Y2))
    
    f1, (ax11, ax12) = plt.subplots(1, 2, figsize=(8, 3))
    ax11.imshow(Y1.reshape(num_rows, num_cols))
    ax12.imshow(pred1.reshape(num_cols, num_cols))

    f2, (ax21, ax22) = plt.subplots(1, 2, figsize=(8, 3))
    ax21.imshow(Y2.reshape(num_rows, num_cols))
    ax22.imshow(pred2.reshape(num_cols, num_cols))
    
    rmse1 = np.linalg.norm(pred1-Y1/y1_max)/np.sqrt(n)
    rmse2 = np.linalg.norm(pred2-Y2/y2_max)/np.sqrt(n)
    print(rmse1, rmse2)

    # # Initialize plots
    # f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
    # test_x = torch.linspace(0, 1, 51)
    # # Make y index vectors of the same length as test data
    # y1_inds_test = torch.zeros(51).long()
    # y2_inds_test = torch.ones(51).long()

    # observed_pred_y1 = gp.predict(test_x, y1_inds_test)
    # observed_pred_y2 = gp.predict(test_x, y2_inds_test)

    # # Define plotting function
    # def ax_plot(ax, train_y, rand_var, title):
    #     # Get lower and upper confidence bounds
    #     lower, upper = rand_var.confidence_region()
    #     # Plot training data as black stars
    #     ax.plot(train_x.data.numpy(), train_y.data.numpy(), 'k*')
    #     # Predictive mean as blue line
    #     ax.plot(test_x.data.numpy(), rand_var.mean().data.numpy(), 'b')
    #     # Shade in confidence 
    #     ax.fill_between(test_x.data.numpy(), lower.data.numpy(), upper.data.numpy(), alpha=0.5)
    #     ax.set_ylim([-3, 3])
    #     ax.legend(['Observed Data', 'Mean', 'Confidence'])
    #     ax.set_title(title)
    # # Plot both tasks
    # ax_plot(y1_ax, train_y1, observed_pred_y1, 'Observed Values (Likelihood)')
    # ax_plot(y2_ax, train_y2, observed_pred_y2, 'Observed Values (Likelihood)')

    plt.show()
    ipdb.set_trace()
    