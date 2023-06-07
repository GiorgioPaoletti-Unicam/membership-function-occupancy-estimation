import numpy as np
import matplotlib.pyplot as plt


def calculate_gaussian_mf_params(u, v, data):
    # u: U-matrix from fuzzy C-means
    # v: cluster centers from fuzzy C-means
    # data: input data

    c, n_features = v.shape

    # Initializing the MF parameters
    mf_params = {}

    for i in range(c):
        # Getting the main data points of the cluster
        main_cluster_points = data[np.argmax(u, axis=1) == i]

        # Calculating mu and sigma
        mu = v[i]  # mean (centroid)
        sigma = np.cov((main_cluster_points - mu).T)  # covariance matrix of distances to the centroid

        mf_params['cluster' + str(i + 1)] = {
            'mean': mu,  # μ
            'std': sigma  # σ
        }

    return mf_params


def calculate_triangular_mf_params(u, v, data, beta):
    # u: U-matrix from fuzzy C-means
    # v: cluster centers from fuzzy C-means
    # data: input data
    # beta: predefined constant

    c, n_features = v.shape

    # Initializing the MF parameters
    mf_params = {}

    for i in range(c):
        # Getting the main data points of the cluster
        main_cluster_points = data[np.argmax(u, axis=1) == i]

        # Calculate alpha and gamma
        alpha = v[i]  # alpha is the centroid of the cluster
        gamma = np.mean(main_cluster_points, axis=0)  # gamma is the mean of the data points in the cluster

        # Calculate a, b, and c for triangular MF
        mf_params['cluster' + str(i + 1)] = {
            'a': alpha - (beta * gamma),
            'b': alpha,
            'c': alpha + (beta * gamma)
        }

    return mf_params


def calculate_trapezoidal_mf_params(u, v, data, beta, delta):
    # u: U-matrix from fuzzy C-means
    # v: cluster centers from fuzzy C-means
    # data: input data
    # beta, delta: parameters beta and delta

    c, n_features = v.shape

    # Initializing the MF parameters
    mf_params = {}

    for i in range(c):
        # Getting the main data points of the cluster
        main_cluster_points = data[np.argmax(u, axis=1) == i]

        # Calculate alpha and gamma
        alpha = v[i]  # alpha is the centroid of the cluster
        gamma = np.mean(main_cluster_points, axis=0)  # gamma is the mean of the data points in the cluster

        # Calculate a, b, c, and d for trapezoidal MF
        mf_params['cluster' + str(i + 1)] = {
            'a': alpha - (beta * gamma),
            'b': alpha - (delta * gamma),
            'c': alpha + (delta * gamma),
            'd': alpha + (beta * gamma)
        }

    return mf_params


def plot_membership_functions(mf_params, n_features, type):
    """
    Plots the membership functions for each feature in each cluster.
    mf_params: parameters of the membership functions
    type: type of the membership function (gaussian, triangular, trapezoidal)

    """
    n_c = len(mf_params)  # Number of clusters
    # n_features = len(mf_params['cluster1'][])  # Number of features

    # Iterate over each feature
    for f in range(n_features):
        plt.figure(figsize=(10, 6))
        plt.title('Feature ' + str(f + 1))

        # Iterate over each cluster
        for i in range(n_c):
            cluster_key = 'cluster' + str(i + 1)

            x, y = None, None
            if type == 'gaussian':
                # Getting the mean and standard deviation for the feature f in cluster i
                mu = mf_params[cluster_key]['mean'][f]
                sigma = np.sqrt(mf_params[cluster_key]['std'][f, f])

                # Plot the Gaussian function
                x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
                y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

            elif type == 'triangular':
                a = mf_params[cluster_key]['a'][f]
                b = mf_params[cluster_key]['b'][f]
                c = mf_params[cluster_key]['c'][f]

                # Plot the Triangular function
                x = np.linspace(a, c, 1000)
                y = np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

            elif type == 'trapezoidal':

                a = mf_params[cluster_key]['a'][f]
                b = mf_params[cluster_key]['b'][f]
                c = mf_params[cluster_key]['c'][f]
                d = mf_params[cluster_key]['d'][f]

                # Plot the Trapezoidal function
                x = np.linspace(a, d, 1000)
                # y = np.maximum(np.minimum(np.minimum((x - a) / (b - a), (d - x) / (d - c)), 1), 0)
                epsilon = 1e-10  # small constant to prevent divide by zero
                y = np.maximum(np.minimum(np.minimum((x - a) / (b - a + epsilon), (d - x) / (d - c + epsilon)), 1), 0)

            plt.plot(x, y, label='Cluster ' + str(i + 1))

        plt.legend()
        plt.show()