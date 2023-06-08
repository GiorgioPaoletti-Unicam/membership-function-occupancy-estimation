import matplotlib.pyplot as plt
import numpy as np


# Abstract class for MF parameters calculator
# @param mf_params: MF parameters
class AbstractPlotter(object):
    def __init__(self, mf_params):
        self._mf_params = mf_params
        self._n_cluster = len(mf_params)
        self._n_features = self._calculate_n_features()

    # Calculate the number of features
    def _calculate_n_features(self):
        raise NotImplementedError("Please Implement this method")

    # Get x and y for plotting
    def _get_x_y(self, cluster_key, f):
        raise NotImplementedError("Please Implement this method")

    # Plot the MFs
    def plot(self):
        plt.figure(figsize=(10, 6 * self._n_features))

        for f in range(self._n_features):
            plt.subplot(self._n_features, 1, f + 1)
            plt.title('Feature ' + str(f + 1))

            # Iterate over each cluster
            for i in range(self._n_cluster):
                cluster_key = 'cluster' + str(i + 1)

                x, y = self._get_x_y(cluster_key, f)
                plt.plot(x, y, label='Cluster ' + str(i + 1))

            plt.legend()

        plt.tight_layout()
        plt.show()


# Plotter for Gaussian MF
class GaussianPlotter(AbstractPlotter):

    def __init__(self, mf_params):
        super().__init__(mf_params)

    def _calculate_n_features(self):
        return len(self._mf_params['cluster1']['mean'])

    def _get_x_y(self, cluster_key, f):
        mu = self._mf_params[cluster_key]['mean'][f]
        sigma = np.sqrt(self._mf_params[cluster_key]['std'][f, f])

        # Plot the Gaussian function
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        y = y / np.max(y)

        return x, y


# Plotter for Triangular MF
class TriangularPlotter(AbstractPlotter):

    def __init__(self, mf_params):
        super().__init__(mf_params)

    # Calculate the number of features for triangular MF
    def _calculate_n_features(self):
        return len(self._mf_params['cluster1']['a'])

    # Get x and y for plotting triangular MF
    def _get_x_y(self, cluster_key, f):
        a = self._mf_params[cluster_key]['a'][f]
        b = self._mf_params[cluster_key]['b'][f]
        c = self._mf_params[cluster_key]['c'][f]

        # Plot the Triangular function
        x = np.linspace(a, c, 1000)
        y = np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

        # Normalize y values to have a range between 0 and 1
        y = y / np.max(y)

        return x, y


# Plotter for Trapezoidal MF
class TrapezoidalPlotter(AbstractPlotter):

    def __init__(self, mf_params):
        super().__init__(mf_params)

    # Calculate the number of features for trapezoidal MF
    def _calculate_n_features(self):
        return len(self._mf_params['cluster1']['a'])

    # Get x and y for plotting trapezoidal MF
    def _get_x_y(self, cluster_key, f):
        a = self._mf_params[cluster_key]['a'][f]
        b = self._mf_params[cluster_key]['b'][f]
        c = self._mf_params[cluster_key]['c'][f]
        d = self._mf_params[cluster_key]['d'][f]

        # Plot the Trapezoidal function
        x = np.linspace(a, d, 1000)
        y = np.maximum(np.minimum(np.minimum((x - a) / (b - a), (d - x) / (d - c)), 1), 0)
        # epsilon = 1e-10  # small constant to prevent divide by zero
        # y = np.maximum(np.minimum(np.minimum((x - a) / (b - a + epsilon), (d - x) / (d - c + epsilon)), 1), 0)

        # Normalize y values to have a range between 0 and 1
        y = y / np.max(y)

        return x, y
