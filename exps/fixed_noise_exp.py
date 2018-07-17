from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF
from env import FieldEnv

import ipdb
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = FieldEnv(num_rows=20, num_cols=20)
    n = 20
    indices = np.random.randint(0, env.num_samples, n)
    noise_std = np.random.rand(n)
    X = env.X[indices, :]
    Y = env.collect_samples(indices, noise_std)

    kernel = RBF(length_scale=1.0)
    model = gaussian_process.GaussianProcessRegressor(
                kernel, alpha=noise_std**2)
    model.fit(X, Y)

    ipdb.set_trace()
    pred, sig = model.predict(env.X, return_std=True)
    
    plt.figure(0)
    plt.title('True values')
    plt.imshow(env.Y.reshape(env.num_rows, env.num_cols))

    plt.figure(1)
    plt.title('Predicted values')
    plt.imshow(pred.reshape(env.num_rows, env.num_cols))

    plt.figure(2)
    plt.title('Variance')
    plt.imshow(sig.reshape(env.num_rows, env.num_cols))

    # plt.show()