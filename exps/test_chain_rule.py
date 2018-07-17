import numpy as np
import ipdb

from utils import generate_gaussian_data, entropy, conditional_entropy
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF


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


if __name__ == '__main__':
    num_rows = 20
    num_cols = 20
    n = num_rows * num_cols
    n_train = 10
    X, Y = generate_gaussian_data(num_rows, num_cols)
    indices_train = np.random.randint(0, n, n_train)
    x_train = X[indices_train, :]
    std = .5
    var = std**2
    y_train = Y[indices_train] + np.random.normal(0, std, n_train)

    krnl = RBF(length_scale=1.0)
    gp = gaussian_process.GaussianProcessRegressor(krnl)
    gp.fit(x_train, y_train)

    n_test = 6
    indices = np.random.randint(0, n, n_test)
    x = X[indices, :]
    joint_ent = entropy(x, gp.kernel_, var)
    ents = []
    for i in range(n_test):
        ents.append(conditional_entropy(x[i], x[0:i], gp.kernel_, var, var))

    print('Joint entropy ', joint_ent)
    print('Sum of conditional entropies ', sum(ents))


