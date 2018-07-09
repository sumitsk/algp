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

# originally in env.py file
# @deprecated
# def select(self, map_pose, max_distance):
#     # NOTE: this is a test I ran to validate the computation of covariance matrix via in-built function and
#     # via actual formula (first principle lets say), see result below
#
#     sensor_unvisited = np.where(self.sensor_visited.flatten() == 0)[0]
#     # sensor_visited = np.where(self.sensor_visited.flatten() == 1)[0]
#     sensor_Y = self.env.X[sensor_unvisited, :]
#     sensor_A = self.sensor_x
#
#     sensor_KYA = self.sensor_model.kernel_(sensor_Y, sensor_A)
#     sensor_KAA = self.sensor_model.kernel_(sensor_A, sensor_A)
#     sensor_KYY = self.sensor_model.kernel_(sensor_Y, sensor_Y)
#
#     # adding sensor_noise to the diagonal elements of KAA satisfies eq2
#     # Possible bug: sometimes off-diagonal elements are not close by
#     sensor_KAA += self.sensor_noise * np.eye(sensor_KAA.shape[0])
#
#     camera_unvisited = np.where(self.camera_visited.flatten() == 0)[0]
#     # camera_visited = np.where(self.camera_visited.flatten() == 1)[0]
#     camera_Y = self.env.X[camera_unvisited, :]
#     camera_A = self.camera_x
#
#     camera_KYA = self.sensor_model.kernel_(camera_Y, camera_A)
#     camera_KAA = self.sensor_model.kernel_(camera_A, camera_A)
#     camera_KYY = self.sensor_model.kernel_(camera_Y, camera_Y)
#
#     # Method1: select location which leads to maximum reduction in entropy
#     mu1, std = self.sensor_model.predict(sensor_Y, return_std=True)
#     mu2, sigma = self.sensor_model.predict(sensor_Y, return_cov=True)
#
#     # verify eq(2) for a vector of inputs
#     # mat1 = sensor_KYY - np.dot(np.dot(sensor_KYA, np.linalg.inv(sensor_KAA)), sensor_KYA.T)
#
#     # RESULT: mat1 is almost always equal to sigma. Sometimes off-diagonal terms are quite off
#
#     # err = np.abs(mat1 - sigma).max()
#     # print(err)
#     # if err > .01:
#     #     ipdb.set_trace()

# @deprecated
# def nearby_locations(self, pose, max_distance):
#     # returns all locations which are less than max distance apart from the current pose
#     # run a BFS search and find all unvisited locations {O(max_distance)}
#
#     locations = self.env.map.get_nearby_locations(pose, max_distance)
#
#     # remove those which have already been sensed
#     ind = np.ravel_multi_index(locations.T, self.sensor_visited.shape)
#     remove_indices = np.where(self.sensor_visited.flatten()[ind] == 1)[0]
#     locations = np.delete(locations, remove_indices, axis=0)
#     return locations


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



