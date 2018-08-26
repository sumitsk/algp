import torch
import gpytorch
import math
from gpytorch.kernels import RBFKernel, IndexKernel
from gpytorch.means import ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable
import matplotlib.pyplot as plt 

from utils import to_torch, to_numpy
import ipdb


class KroneckerMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_tasks):
        super(KroneckerMultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        self.n_tasks = n_tasks
        self.mean_module = ZeroMean()
        self.covar_module = RBFKernel(ard_num_dims=train_x.size(-1))
        self.task_covar_module = IndexKernel(n_tasks=n_tasks, rank=1)
        self.register_buffer('task_indices', torch.LongTensor([0, 1]))

    def forward(self, x):
        mean_x = self.mean_module(x).repeat(self.n_tasks)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(self.task_indices)
        # The covariance matrix that we use is the kronecker product between the input and task covar matrices
        # Here we use a "KroneckerProductLazyVariable" to do the kronecker product
        # This prevents us from actually computing the complete matrix, and we instead get major computation savings
        covar_xi = gpytorch.lazy.KroneckerProductLazyVariable(covar_i, covar_x)
        return GaussianRandomVariable(mean_x, covar_xi)


# class HadamardMultitaskGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, n_tasks):
#         super(HadamardMultitaskGPModel, self).__init__(train_x, train_y, likelihood)

#         self.n_tasks = n_tasks
#         self.mean_module = ZeroMean()
#         self.covar_module = RBFKernel()
#         self.task_covar_module = IndexKernel(n_tasks=n_tasks, rank=1)

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         covar_i = self.task_covar_module(self.task_indices)
#         covar_xi = covar_x.mul(covar_i)
#         return GaussianRandomVariable(mean_x, covar_xi)


class MultitaskGP(object):
    def __init__(self, n_tasks, lr=.1, max_iter=100):
        super(MultitaskGP, self).__init__()
        self.n_tasks = n_tasks
        self.lr = lr
        self.max_iter = max_iter
        
    def reset(self, x, ys):
        self._train_x = to_torch(x)
        assert len(ys) == self.n_tasks
        self._train_ys = [to_torch(y_) for y_ in ys]
        self._train_ys_mean = [y.mean() for y in self._train_ys]
        self._norm_train_y = torch.cat([y - mean_y for y, mean_y in zip(self._train_ys, self._train_ys_mean)])

        self.likelihood = GaussianLikelihood()
        self.model = KroneckerMultitaskGPModel(self._train_x, self._norm_train_y, self.likelihood, self.n_tasks)
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def fit(self, x, ys):
        self.reset(x, ys)
        self.model.train()
        self.likelihood.train()
        for i in range(self.max_iter):
            self.optimizer.zero_grad()
            output = self.model(self._train_x)
            loss = -self.mll(output, self._norm_train_y)
            loss.backward()
            self.optimizer.step()
            # print(i, loss.item())
            if i == 0:
                initial_ll = -loss.item()
            elif i == self.max_iter - 1:
                final_ll = -loss.item()

        print('Initial LogLikelihood {:.3f} Final LogLikelihood {:.3f}'.format(initial_ll, final_ll))


    def predict(self, x):
        self.model.eval()
        self.likelihood.eval()
        x_ = to_torch(x)
        with gpytorch.fast_pred_var() and torch.no_grad():
            pred_rand_var = self.likelihood(self.model(x_))
            pred_means = torch.chunk(pred_rand_var.mean(), self.n_tasks)
            pred_means = [m + mean_y for m, mean_y in zip(pred_means, self._train_ys_mean)]
        return pred_means, pred_rand_var

# Define plotting function
def ax_plot(ax, x, y, pred_rand_var, title):
    # Get lower and upper confidence bounds
    lower, upper = pred_rand_var.confidence_region()
    ax.plot(x, y, 'k*')
    ax.plot(x, pred_rand_var.mean().data.numpy(), 'b')
    ax.fill_between(x, lower.data.numpy(), upper.data.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(title)


if __name__ == '__main__':
    train_x = torch.linspace(0, 1, 100)

    train_y1 = torch.sin(train_x.data * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
    train_y2 = torch.cos(train_x.data * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
    
    gp = MultitaskGP(n_tasks=2, lr=.05)
    gp.fit(train_x, [train_y1, train_y2])
    test_x = torch.linspace(0, 1, 101)
    # contains prediction for both tasks
    pred_means, pred_rand_var = gp.predict(test_x)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    trx, tex, y1, y2 = to_numpy(train_x), to_numpy(test_x), to_numpy(train_y1), to_numpy(train_y2)
    lower, upper = pred_rand_var.confidence_region()
    lbs = torch.chunk(lower, gp.n_tasks)
    ubs = torch.chunk(upper, gp.n_tasks)
    ipdb.set_trace()

    ax1.plot(trx, y1, 'k*')
    ax1.plot(tex, to_numpy(pred_means[0]), 'b')
    ax1.fill_between(tex, to_numpy(lbs[0]), to_numpy(ubs[0]), alpha=.5)
    ax1.set_ylim([-3, 3])

    ax2.plot(trx, y2, 'k*')
    ax2.plot(tex, to_numpy(pred_means[1]), 'b')
    ax2.fill_between(tex, to_numpy(lbs[1]), to_numpy(ubs[1]), alpha=.5)
    ax2.set_ylim([-3, 3])
    plt.show()