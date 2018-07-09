import numpy as np
import ipdb

from utils import generate_gaussian_data, conditional_entropy, mutual_information
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF
from utils import conditional_entropy

def get_measurement(indices, noise_std, Y):
    y = Y[indices] + np.random.normal(0, noise_std, size=len(indices))
    return y


def entropy(y, A, model, noise_std):
    # ipdb.set_trace()
    kernel = model.kernel_
    dim = len(y)
    y = y.reshape(-1, dim)
    A = A.reshape(-1, dim)
    n = y.shape[0]

    sigma_AA = kernel(A, A) + noise_std**2 * np.eye(A.shape[0])
    sigma_yA = kernel(y, A)
    cov1 = kernel(y, y) - np.dot(np.dot(sigma_yA, np.linalg.inv(sigma_AA)), sigma_yA.T)
    ent1 = .5*np.log(((2*np.pi*np.exp(1))**n)*np.linalg.det(cov1))
    # mu1 =

    ent2 = conditional_entropy(y, model)

    return ent2


if __name__ == '__main__':
    num_rows = 20
    num_cols = 20
    n = num_rows * num_cols
    sampled = np.zeros(n)
    noise_std = 0.1
    X, Y = generate_gaussian_data(num_rows, num_cols)

    krnl = RBF(length_scale=1.0)
    model = gaussian_process.GaussianProcessRegressor(krnl, alpha=noise_std**2)
    indices = np.random.randint(low=0, high=n, size=10)
    sampled[indices] = 1
    model.fit(X[indices], get_measurement(indices, noise_std, Y))

    # verify the chain rule of entropy
    ind = np.random.randint(low=0, high=n, size=2)
    # e1 = entropy(X[ind[0], :], X[indices], model, noise_std)
    # e2 = entropy(X[ind[1], :], X[np.array(list(indices) + [ind[0]])], model, noise_std)
    # e3 = entropy(X[ind, :], X[indices, :], model, noise_std)

    e1 = conditional_entropy(X[ind, :], model)
    mi = mutual_information(X[ind, :], model)

    ipdb.set_trace()
    # print(e1 + e2 - e3)



