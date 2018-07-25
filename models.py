from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, Matern

import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from gpytorch.kernels import RBFKernel, WhiteNoiseKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

import numpy as np
import ipdb
from utils import to_torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class SklearnGPR(object):
    def __init__(self, kernel=None):
        if kernel == 'rbf' or kernel is None:
            self.init_kernel = RBF(length_scale=1.0)
        elif kernel == 'matern':
            self.init_kernel = Matern(nu=1.5, length_scale=1.0)
        else:
            raise NotImplementedError
        self.model = None

    @property
    def train_x(self):
        return self.model.X_train_

    @property
    def train_var(self):
        return self.model.alpha

    @property
    def kernel_(self):
        return self.model.kernel_
        
    def set_train_var(self, var):
        self.model.alpha = var

    def fit(self, x, y, var=None):
        self.reset()
        self.set_train_var(var if var is not None else 1e-10)
        self.model.fit(x, y)

    def predict(self, x, return_std=False, return_cov=False):
        assert not (return_std and return_cov), 'Can return either std or var'
        return self.model.predict(x, return_std=return_std, return_cov=return_cov)

    def reset(self):
        self.model = gaussian_process.GaussianProcessRegressor(self.init_kernel)

    def cov_mat(self, x1, x2, white_noise=None):
        cov = self.model.kernel_(x1, x2)
        if white_noise is not None:
            cov = cov + white_noise
        return cov

    
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, var=None, latent=None, **kwargs):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.rbf_covar_module = RBFKernel()
        
        # set covariance module
        if var is not None:
            self.noise_covar_module = WhiteNoiseKernel(var)
            self.covar_module = self.rbf_covar_module + self.noise_covar_module
        else:
            self.covar_module = self.rbf_covar_module
        self._set_latent_function(latent, **kwargs) 

    def _set_latent_function(self, latent, **kwargs):
        if latent is None or latent == 'identity':
            self.latent_func = IdentityLatentFunction()
        elif latent == 'linear':
            self.latent_func = LinearLatentFunction(**kwargs)
        elif latent == 'non_linear':
            self.latent_func = NonLinearLatentFunction(**kwargs)
        elif latent == 'field':
            self.latent_func = FieldLatentFunction(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, inp):
        x = self.latent_func(inp)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


class IdentityLatentFunction(nn.Module):
    def __init__(self):
        super(IdentityLatentFunction, self).__init__()
        self.apply(weights_init)

    def forward(self, x):
        return x


class LinearLatentFunction(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(LinearLatentFunction, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)
        # self.apply(weights_init)

    def forward(self, x):
        return self.fc(x)


class NonLinearLatentFunction(nn.Module):
    def __init__(self, input_dim):
        super(NonLinearLatentFunction, self).__init__()
        f1_dim = 2
        f2_dim = 2
        self.fc1 = nn.Linear(input_dim, f1_dim)
        self.fc2 = nn.Linear(f1_dim, f2_dim)
        # self.apply(weights_init)

    def forward(self, inp):
        x = F.relu(self.fc1(inp))
        x = self.fc2(x)
        return x


class FieldLatentFunction(nn.Module):
    def __init__(self, rr_dim, v_dim):
        super(FieldLatentFunction, self).__init__()
        self.rr_dim = rr_dim
        self.v_dim = v_dim
        f1_dim = 3
        f2_dim = 3
        # NOTE: it may not be a good idea to perform transformation of rr dimensions
        self.rr_fc = nn.Linear(self.rr_dim, f1_dim)
        self.v_fc = nn.Linear(self.v_dim, f1_dim)
        self.fc = nn.Linear(2*f1_dim, f2_dim)
        # self.apply(weights_init)

    def forward(self, inp):
        # TODO: perhaps too many parameters (might overfit)
        inp1, inp2 = torch.split(inp, [self.rr_dim, self.v_dim], dim=-1)
        x1 = F.relu(self.rr_fc(inp1))
        x2 = F.relu(self.v_fc(inp2))
        x = torch.cat([x1, x2], dim=-1)
        x = self.fc(x)
        return x


# TODO: try batchnorm and lr scheduler
class GpytorchGPR(object):
    def __init__(self, latent=None, lr=.1):
        self._train_x = None
        self._train_y = None
        self._train_var = None
        self.likelihood = None
        self.model = None
        self.optimizer = None
        self.mll = None
        self.lr = lr
        self.latent = latent
        self.latent_func = None

    @property
    def train_x(self):
        return self._train_x.cpu().numpy()

    @property
    def train_var(self):
        return self._train_var.cpu().numpy()

    def reset(self, x, y, var, **kwargs):
        self._train_x = torch.FloatTensor(x)
        self._train_y = torch.FloatTensor(y)
        if var is not None:
            self._train_var = torch.FloatTensor(var)

        self.likelihood = GaussianLikelihood()
        self.model = ExactGPModel(self._train_x, self._train_y,
                                  self.likelihood, self._train_var,
                                  self.latent, **kwargs)

        self.optimizer = torch.optim.Adam(
            [{'params': self.model.parameters()}, ], lr=self.lr)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)

    def fit(self, x, y, var=None, **kwargs):
        self.reset(x, y, var, **kwargs)
        self.model.train()
        self.likelihood.train()

        training_iter = 1000
        last_loss = 0
        loss_change = []
        # TODO: High priority (implement a robust convergence condition)
        for i in range(training_iter):
            self.optimizer.zero_grad()
            output = self.model(self._train_x)
            loss = -self.mll(output, self._train_y)/len(y)
            loss.backward()
            self.optimizer.step()
            delta = loss.item() - last_loss
            # loss_change.append(abs(delta))
            # if np.mean(loss_change[max(0, i-10):i+1]) < .01:
            #     break
            last_loss = loss.item()
            print(i, last_loss)
        # ipdb.set_trace()

    def cov_mat(self, x1, x2, white_noise=None):
        x1_ = to_torch(x1)
        x2_ = to_torch(x2)
        # important to set model to eval mode
        self.model.eval()
        with torch.no_grad():
            cov = self.model.covar_module(x1_, x2_).evaluate().cpu().numpy()
        # add white noise component here
        if white_noise is not None:
            cov = cov + white_noise
        return cov

    def predict(self, x, return_cov=True, return_std=False):
        self.model.eval()
        self.likelihood.eval()
        x_ = to_torch(x)

        with gpytorch.fast_pred_var() and torch.no_grad():
            pred = self.likelihood(self.model(x_))
            pred_mean = pred.mean().cpu().numpy()
            if return_std:
                std = pred.std().cpu().numpy()
                # ipdb.set_trace()
                return pred_mean, std
            elif return_cov:
                cov = pred.covar().evaluate().cpu().numpy()
                return pred_mean, cov
            else:
                return pred_mean
