import math
import torch
import gpytorch
from matplotlib import pyplot as plt

from gpytorch.kernels import RBFKernel, SpectralMixtureKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable
import ipdb

# Training data points are located every 0.075 along 0 to 0.75 inclusive
train_x = torch.linspace(0, 1, 15)
train_y = torch.sin(train_x.data * (2 * math.pi))
train_x = torch.rand(train_x.size(0), 2)
ipdb.set_trace()

# True function is sin(2*pi*x)
# Gaussian noise N(0,0.04) added

# Here we see an example of using the spectral mixture kernel as described here:
# https://arxiv.org/pdf/1302.4245.pdf
class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        # We can learn a mean between -1 and 1
        self.mean_module = ConstantMean()
        # We use a spectral mixture kernel where the frequency is a mixture of 3 Gaussians
        self.covar_module = SpectralMixtureKernel(n_mixtures=4, n_dims=train_x.size(-1))
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)

# Initialize the likelihood. We use a Gaussian for regression to get predictive mean
# and variance and learn noise parameter
likelihood = GaussianLikelihood()
# Use the likelihood to initialize the model
model = SpectralMixtureGPModel(train_x.data, train_y.data, likelihood)


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 100
for i in range(training_iter):
    # Zero previously backpropped gradients
    optimizer.zero_grad()
    # Make prediction
    output = model(train_x)
    # Calc loss and backprop
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.data[0]))
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()

# Initialize figure
f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
# Test points every 0.1 between 0 and 5
# (note this is over 6 times the length of the region with training points)
test_x = torch.linspace(0, 2, 51)
test_x = torch.cat([test_x.unsqueeze(-1), torch.ones_like(test_x.unsqueeze(-1))], dim=-1)

# Make predictions
observed_pred = likelihood(model(test_x))

# Define plotting function
def ax_plot(ax, rand_var, title):
    # Get lower and upper confidence bounds
    lower, upper = rand_var.confidence_region()
    # Training data as black stars
    ax.plot(train_x.data.numpy(), train_y.data.numpy(), 'k*')
    # Plot predictive mean as blue line
    ax.plot(test_x.data.numpy(), rand_var.mean().data.numpy(), 'b')
    # Shade confidence region
    ax.fill_between(test_x.data.numpy(), lower.data.numpy(), upper.data.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    # Labels + title
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(title)
# Plot figure
# ax_plot(observed_ax, observed_pred, 'Observed Values (Likelihood)')


plt.show()