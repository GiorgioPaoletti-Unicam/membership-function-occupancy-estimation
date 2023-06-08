import numpy as np


# Abstract class for MF parameters calculator
# @param name: name of the MF
# @param u: U-matrix from fuzzy C-means
# @param v: cluster centers from fuzzy C-means
# @param data: input data
class AbstractFMParamsCalculator(object):
    def __init__(self, u, v, data):
        self._u = u
        self._v = v
        self._data = data
        self._c, self._n_features = v.shape
        self._mf_params = {}

    def calculate(self):
        raise NotImplementedError("Please Implement this method")


# Implementing the Gaussian MF parameters calculator
class GaussianFMParamsCalculator(AbstractFMParamsCalculator):

    def __init__(self, u, v, data):
        super().__init__(u, v, data)

    def calculate(self):
        for i in range(self._c):
            # Getting the main data points of the cluster
            main_cluster_points = self._data[np.argmax(self._u, axis=1) == i]

            # Calculating mu and sigma
            mu = self._v[i]  # mean (centroid)
            sigma = np.cov((main_cluster_points - mu).T)  # covariance matrix of distances to the centroid

            self._mf_params['cluster' + str(i + 1)] = {
                'mean': mu,  # μ
                'std': sigma  # σ
            }

        return self._mf_params


# Implementing the triangular MF parameters calculator
# @param beta: predefined constant
class TriangularFMParamsCalculator(AbstractFMParamsCalculator):
    def __init__(self, u, v, data, beta):
        super().__init__(u, v, data)
        self._beta = beta

    def calculate(self):
        for i in range(self._c):
            # Getting the main data points of the cluster
            main_cluster_points = self._data[np.argmax(self._u, axis=1) == i]

            # Calculate alpha and gamma
            alpha = self._v[i]  # alpha is the centroid of the cluster
            gamma = np.mean(main_cluster_points, axis=0)  # gamma is the mean of the data points in the cluster

            # Calculate a, b, and c for triangular MF
            self._mf_params['cluster' + str(i + 1)] = {
                'a': alpha - (self._beta * gamma),
                'b': alpha,
                'c': alpha + (self._beta * gamma)
            }

        return self._mf_params


# Implementing the trapezoidal MF parameters calculator
# @param beta, delta: parameters beta and delta
class TrapezoidalFMParamsCalculator(AbstractFMParamsCalculator):
    def __init__(self, u, v, data, beta, delta):
        super().__init__(u, v, data)
        self._beta = beta
        self._delta = delta

    def calculate(self):
        for i in range(self._c):
            # Getting the main data points of the cluster
            main_cluster_points = self._data[np.argmax(self._u, axis=1) == i]

            # Calculate alpha and gamma
            alpha = self._v[i]  # alpha is the centroid of the cluster
            gamma = np.mean(main_cluster_points, axis=0)  # gamma is the mean of the data points in the cluster

            # Calculate a, b, c, and d for trapezoidal MF
            # TODO: A of the HumidityRatio is always negative --> the plot will plot a triangle instead of a trapezoid
            self._mf_params['cluster' + str(i + 1)] = {
                'a': alpha - (self._beta * gamma) - self._delta,
                'b': alpha - (self._beta * gamma),
                'c': alpha + (self._beta * gamma),
                'd': alpha + (self._beta * gamma) + self._delta
            }

        return self._mf_params
