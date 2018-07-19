from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, Matern

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, IndexKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

import ipdb


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
    def kernel(self):
        return self.model.kernel_

    def set_train_var(self, var):
        self.model.alpha = var

    def fit(self, x, y, var):
        self.reset()
        self.set_train_var(var)
        self.model.fit(x, y)

    def predict(self, x, return_std=False, return_cov=False):
        assert not (return_std and return_cov), 'Can return either std or var'
        return self.model.predict(x, return_std=return_std, return_cov=return_cov)

    def reset(self):
        self.model = gaussian_process.GaussianProcessRegressor(self.init_kernel)


class ExactManifoldGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, embed_dim=4):
        super(ExactManifoldGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-5, 5))
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
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-5, 5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


class GpytorchGPR(object):
    def __init__(self, use_embed=False):
        self.use_embed = use_embed
        self.train_x = None
        self.train_y = None
        self.likelihood = None
        self.model = None
        self.optimizer = None
        self.mll = None

    # @property
    # def train_x(self):
    #     return self._train_x

    def train_var(self):
        pass

    def reset(self, tx, ty, var):
        # ipdb.set_trace()
        # torch.manual_seed(0)
        self.likelihood = GaussianLikelihood(log_noise_bounds=(-5, 5))
        if self.use_embed:
            self.model = ExactManifoldGPModel(tx.data, ty.data, self.likelihood)
        else:
            self.model = ExactGPModel(tx.data, ty.data, self.likelihood)

        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}, ], lr=0.1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.train_x = tx
        self.train_y = ty

    def fit(self, x, y, var=None):
        tx = torch.FloatTensor(x)
        ty = torch.FloatTensor(y)
        self.reset(tx, ty, var)
        self.model.train()
        self.likelihood.train()

        training_iter = 500
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
        