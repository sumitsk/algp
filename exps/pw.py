from env import FieldEnv
from models import GpytorchGPR
import numpy as np 
import ipdb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def draw_plots(num_rows, num_cols, true_y, pred_y, pred_var):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    axt, axp, axv = ax
    
    axt.set_title('Ground truth')
    imt = axt.imshow(true_y.reshape(num_rows, num_cols), cmap='ocean')
    div = make_axes_locatable(axt)
    caxt = div.new_horizontal(size='5%', pad=.05)
    fig.add_axes(caxt)
    fig.colorbar(imt, caxt, orientation='vertical')

    axp.set_title('Predicted values')
    imp = axp.imshow(pred_y.reshape(num_rows, num_cols), cmap='ocean')
    divm = make_axes_locatable(axp)
    caxp = divm.new_horizontal(size='5%', pad=.05)
    fig.add_axes(caxp)
    fig.colorbar(imp, caxp, orientation='vertical')

    axv.set_title('Variance')
    imv = axv.imshow(pred_var.reshape(num_rows, num_cols), cmap='hot')
    divv = make_axes_locatable(axv)
    caxv = divv.new_horizontal(size='5%', pad=.05)
    fig.add_axes(caxv)
    fig.colorbar(imv, caxv, orientation='vertical')
    plt.show()


def transform_y(y, inverse=False):
    # return y
    if not inverse:
        return np.log(y + 1)
    return np.exp(y) - 1

# file with field data
data_file = 'data/plant_width_mean_dataset.pkl'
env = FieldEnv(data_file=data_file)
gp = GpytorchGPR(use_embed=True)

n = env.num_samples
num_train = int(.7*n)
train_indices = np.random.randint(0, n, num_train)
test_indices = np.array(list(set(range(n)) - set(train_indices)))

noise_std = .05
noise_var = np.full(num_train, noise_std**2)
x_train = env.X[train_indices, :] 
y_train = env.collect_samples(train_indices, noise_std)
y_train = transform_y(y_train)
x_test = env.X[test_indices, :]

gp.fit(x_train, y_train, noise_var)
mu, std = gp.predict(env.X, return_std=True)
mu = transform_y(mu, True)

rmse = np.linalg.norm(mu-env.Y) / np.sqrt(n)
print('RMSE ', rmse)
# check sklearn and gpytorch variance estimation
draw_plots(env.num_rows, env.num_cols, env.Y, mu, std**2)