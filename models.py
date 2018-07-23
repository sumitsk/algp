from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, Matern

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, WhiteNoiseKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

import numpy as np
import ipdb
from utils import to_torch


class SklearnGPR(object):
    def __init__(self):
        self.init_kernel = RBF(length_scale=1.0)
        self.model = None
        # model is initialised in the reset method
        # self.reset()
        
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


class ExactManifoldGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, embed_dim=4, var=None):
        super(ExactManifoldGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.rbf_covar_module = RBFKernel(log_lengthscale_bounds=(-5, 5))
        if var is not None:
            self.noise_covar_module = WhiteNoiseKernel(var)
            self.covar_module = self.rbf_covar_module + self.noise_covar_module
        else:
            self.covar_module = self.rbf_covar_module

        feature_dim = train_x.size(0)
        self.fc = torch.nn.Linear(feature_dim, embed_dim)

    def _fwd(self, x):
        return self.fc(x)

    def forward(self, x):
        x = self._fwd(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)
    
    
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, var=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.rbf_covar_module = RBFKernel()
        if var is not None:
            self.noise_covar_module = WhiteNoiseKernel(var)
            self.covar_module = self.rbf_covar_module + self.noise_covar_module
        else:
            self.covar_module = self.rbf_covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


class GpytorchGPR(object):
    def __init__(self, use_embed=False):
        self.use_embed = use_embed
        self._train_x = None
        self._train_y = None
        self._train_var = None
        self.likelihood = None
        self.model = None
        self.optimizer = None
        self.mll = None

    @property
    def train_x(self):
        return self._train_x.cpu().numpy()

    @property
    def train_var(self):
        return self._train_var.cpu().numpy()

    def reset(self, x, y, var):
        self._train_x = torch.FloatTensor(x)
        self._train_y = torch.FloatTensor(y)
        if var is not None:
            self._train_var = torch.FloatTensor(var)

        self.likelihood = GaussianLikelihood()
        if self.use_embed:
            self.model = ExactManifoldGPModel(
                self._train_x, self._train_y, self.likelihood,
                self._train_var)
        else:
            self.model = ExactGPModel(
                self._train_x, self._train_y, self.likelihood,
                self._train_var)
        self.optimizer = torch.optim.Adam(
            [{'params': self.model.parameters()}, ], lr=0.1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)

    def fit(self, x, y, var=None):
        self.reset(x, y, var)
        self.model.train()
        self.likelihood.train()

        training_iter = 100
        last_loss = 0
        loss_change = []
        for i in range(training_iter):
            self.optimizer.zero_grad()
            output = self.model(self._train_x)
            loss = -self.mll(output, self._train_y)
            loss.backward()
            self.optimizer.step()
            delta = loss.item() - last_loss
            loss_change.append(abs(delta))
            if np.mean(loss_change[max(0, i-10):i+1]) < .01:
                break
            last_loss = loss.item()

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
                return pred_mean, std
            elif return_cov:
                cov = pred.covar().evaluate().cpu().numpy()
                return pred_mean, cov
            else:
                return pred_mean
