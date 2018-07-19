import math
import torch
import gpytorch
from matplotlib import pyplot as plt

from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.random_variables import GaussianRandomVariable

import ipdb

assert torch.__version__ == '0.4.0', 'Use Pytorch 0.4.0'


# Define plotting function
def ax_plot(ax, rand_var, title):
    # Get upper and lower confidence bounds
    lower, upper = rand_var.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.data.numpy(), train_y.data.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.data.numpy(), rand_var.mean().data.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.data.numpy(), lower.data.numpy(), upper.data.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(title)


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Our mean function is constant in the interval [-1,1]
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        # We use the RBF kernel as a universal approximator
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-5, 5))
    
    def forward(self, x):
        # x = x * 10
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Return model output as GaussianRandomVariable
        return GaussianRandomVariable(mean_x, covar_x)


class ExactGP(object):
    def __init__(self, use_embed=False):
        self.use_embed = use_embed
        self.likelihood = None
        self.model = None
        self.optimizer = None
        self.mll = None
        self.train_x = None
        self.train_y = None

    def reset(self, train_x, train_y):
        self.likelihood = GaussianLikelihood(log_noise_bounds=(-5, 5))
        if self.use_embed:
            self.model = ManifoldExactGPModel(train_x.data, train_y.data, self.likelihood)
        else:
            self.model = ExactGPModel(train_x.data, train_y.data, self.likelihood)

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=0.1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.train_y = train_y
        self.train_x = train_x

    def fit(self, train_x, train_y):
        self.reset(train_x, train_y)
        self.model.train()
        self.likelihood.train()

        training_iter = 100
        for i in range(training_iter):
            self.optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -self.mll(output, self.train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
                i + 1, training_iter, loss.data[0],
                self.model.covar_module.log_lengthscale.data[0, 0],
                self.model.likelihood.log_noise.data[0]
            ))
            self.optimizer.step()

    def test(self, test_x):
        self.model.eval()
        self.likelihood.eval()

        with gpytorch.fast_pred_var() and torch.no_grad():
            pred = self.likelihood(self.model(test_x))

        f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
        ax_plot(observed_ax, pred, 'Observed Values (Likelihood)')    
        plt.show()


class ManifoldExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_dim=1):
        super(ManifoldExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-5, 5))
        embed_dim = 4
        self.fc = torch.nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.fc(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


if __name__ == '__main__':
    train_x = torch.linspace(0, 1, 11)
    train_y = torch.sin(train_x.data * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2

    import numpy as np
    x = np.load('gpx.npy')
    y = np.load('gpy.npy')

    ipdb.set_trace()

    torch.manual_seed(1)
    from models import GpytorchGPR
    gp = GpytorchGPR()
    gp.fit(x, y)

    # train_x = torch.FloatTensor(x)
    # train_y = torch.FloatTensor(y)

    # gp1 = ExactGP(use_embed=False)
    # gp1.fit(train_x, train_y)
    # p1 = [param for param in gp1.model.named_parameters()]
    # test_x = torch.linspace(0, 1, 51)
    # gp1.test(test_x)
    # gp2.test(test_x)

