import math
import torch
import gpytorch
from matplotlib import pyplot as plt

from gpytorch.kernels import RBFKernel, IndexKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

import ipdb


# Define plotting function
def ax_plot(ax, train_y, rand_var, title):
    # Get lower and upper confidence bounds
    lower, upper = rand_var.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.data.numpy(), train_y.data.numpy(), 'k*')
    # Predictive mean as blue line
    ax.plot(test_x.data.numpy(), rand_var.mean().data.numpy(), 'b')
    # Shade in confidence 
    ax.fill_between(test_x.data.numpy(), lower.data.numpy(), upper.data.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(title)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-3, 3))
        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = IndexKernel(n_tasks=2, rank=1)

    def forward(self,x,i):
        mean_x = self.mean_module(x)
        # Get all covariances, we'll look up the task-speicific ones
        covar_x = self.covar_module(x)
        # Get the covariance for task i
        covar_i = self.task_covar_module(i)
        covar_xi = covar_x.mul(covar_i)
        # ipdb.set_trace()
        return GaussianRandomVariable(mean_x, covar_xi)


class MultitaskGP(object):
    def __init__(self, train_x, train_y1, train_y2, y1_inds, y2_inds):
        self.train_x = train_x
        self.train_y1 = train_y1
        self.train_y2 = train_y2
        self.y1_inds = y1_inds
        self.y2_inds = y2_inds

        x = (torch.cat([train_x.data, train_x.data]),
             torch.cat([y1_inds.data, y2_inds.data]))
        y = torch.cat([train_y1.data, train_y2.data])
        self.likelihood = GaussianLikelihood(log_noise_bounds=(-6, 6))
        self.model = MultitaskGPModel(x, y, self.likelihood)

        # includes gaussian likelihood parameters
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},], lr=0.1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def fit(self):
        self.model.train()
        self.likelihood.train()

        iterations = 50
        for i in range(iterations):
            self.optimizer.zero_grad()
            # Make predictions from training data
            # Again, note feeding duplicated x_data and indices indicating which task
            x = torch.cat([self.train_x, self.train_x])
            idx = torch.cat([self.y1_inds, self.y2_inds])
            y = torch.cat([self.train_y1, self.train_y2])
            output = self.model(x, idx)
            # Calc the loss and backprop gradients
            loss = -self.mll(output, y)
            loss.backward()
            print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.data[0]))
            self.optimizer.step()

    def test(self, test_x, y1_inds_test, y2_inds_test):
        self.model.eval()
        self.likelihood.eval()

        pred_y1 = self.likelihood(self.model(test_x, y1_inds_test))
        pred_y2 = self.likelihood(self.model(test_x, y2_inds_test))

        # initialise plots
        f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
        ax_plot(y1_ax, train_y1, pred_y1, 'Observed Values (Likelihood)')
        ax_plot(y2_ax, train_y2, pred_y2, 'Observed Values (Likelihood)')
        plt.show()


if __name__ == '__main__':
    train_x = torch.linspace(0, 1, 11)
    # y1s are indexed 0, y2s are indexed 1
    y1_inds = torch.zeros(11).long()
    y2_inds = torch.ones(11).long()

    train_y1 = torch.sin(train_x.data * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
    train_y2 = torch.cos(train_x.data * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2

    gp_model = MultitaskGP(train_x, train_y1, train_y2, y1_inds, y2_inds)
    gp_model.fit()
    test_x = torch.linspace(0, 1, 51)
    y1_inds_test = torch.zeros(51).long()
    y2_inds_test = torch.ones(51).long()
    gp_model.test(test_x, y1_inds_test, y2_inds_test)
