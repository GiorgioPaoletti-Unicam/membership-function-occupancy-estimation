import matplotlib.pyplot as plt
import numpy as np


# Abstract class for MF parameters calculator
# @param mf_params: MF parameters
class AbstractPlotter(object):
    def __init__(self, mf_params, input_universes):
        self._mf_params = mf_params
        self._input_universes = input_universes
        self._n_cluster = len(mf_params)
        self._n_features = self._calculate_n_features()
        self._features_name = list(input_universes.keys())

    # Calculate the number of features
    def _calculate_n_features(self):
        raise NotImplementedError("Please Implement this method")

    # Get x and y for plotting
    def _get_x_y(self, cluster_key, f, feature):
        raise NotImplementedError("Please Implement this method")

    # Plot the MFs
    def plot(self):
        plt.figure(figsize=(10, 6 * self._n_features))

        # for f in range(self._n_features):
        for f, feature in enumerate(self._features_name):
            plt.subplot(self._n_features, 1, f + 1)
            # plt.title('Feature ' + str(f + 1))
            plt.title(self._features_name[f])

            # Iterate over each cluster
            for i in range(self._n_cluster):
                cluster_key = 'cluster' + str(i + 1)

                x, y = self._get_x_y(cluster_key, f, feature)
                plt.plot(x, y, label='Cluster ' + str(i + 1))

            plt.legend()

        plt.tight_layout()
        plt.show()


# Plotter for Gaussian MF
class GaussianPlotter(AbstractPlotter):
    def __init__(self, mf_params, input_universes):
        super().__init__(mf_params, input_universes)

    def _calculate_n_features(self):
        return len(self._mf_params['cluster1']['mean'])

    def _get_x_y(self, cluster_key, f, feature):
        mu = self._mf_params[cluster_key]['mean'][f]
        sigma = np.sqrt(self._mf_params[cluster_key]['std'][f, f])

        min = mu - 3 * sigma
        max = mu + 3 * sigma

        if mu - 3 * sigma < 0:
            # min = 0
            min = self._input_universes[feature].min()

        if mu + 3 * sigma > self._input_universes[feature].max():
            max = self._input_universes[feature].max()

        x = np.linspace(min, max, 1000)

        """min_val = max(self._input_universes[feature].min(), mu - 3 * sigma)
        max_val = min(mu + 3 * sigma, self._input_universes[feature].max())

        x = np.linspace(min_val, max_val, 1000)"""

        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        y = y / np.max(y)
        return x, y


# Plotter for Triangular MF
class TriangularPlotter(AbstractPlotter):
    def __init__(self, mf_params, input_universes):
        super().__init__(mf_params, input_universes)

    # Calculate the number of features for triangular MF
    def _calculate_n_features(self):
        return len(self._mf_params['cluster1']['a'])

    # Get x and y for plotting triangular MF
    def _get_x_y(self, cluster_key, f, feature):
        a = self._mf_params[cluster_key]['a'][f]
        b = self._mf_params[cluster_key]['b'][f]
        c = self._mf_params[cluster_key]['c'][f]
        x = np.linspace(a, c, 1000)
        y = np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)
        y = y / np.max(y)
        return x, y


# Plotter for Trapezoidal MF
class TrapezoidalPlotter(AbstractPlotter):
    def __init__(self, mf_params, input_universes):
        super().__init__(mf_params, input_universes)

    # Calculate the number of features for trapezoidal MF
    def _calculate_n_features(self):
        return len(self._mf_params['cluster1']['a'])

    # Get x and y for plotting trapezoidal MF
    def _get_x_y(self, cluster_key, f, feature):
        a = self._mf_params[cluster_key]['a'][f]
        b = self._mf_params[cluster_key]['b'][f]
        c = self._mf_params[cluster_key]['c'][f]
        d = self._mf_params[cluster_key]['d'][f]
        x = np.linspace(a, d, 1000)
        y = np.maximum(np.minimum(np.minimum((x - a) / (b - a), (d - x) / (d - c)), 1), 0)
        y = y / np.max(y)
        return x, y
