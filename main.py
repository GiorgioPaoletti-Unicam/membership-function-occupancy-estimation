import pandas as pd

from membership_functions import *

num_clusters = 3
n_features = 5


def read_csv_data(filename):
    # Read data from a CSV file
    data = pd.read_csv(filename, usecols=['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])

    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data


"""In the function above, the fuzzy C-means (FCM) algorithm is implemented. The FCM is an iterative algorithm that 
clusters data by updating membership matrix and cluster centers until convergence. It is a soft clustering method, 
which means that each data point has a degree of belonging to clusters, rather than belonging completely to just one 
cluster. This is why a U-matrix is used to store the membership of each data point to each cluster."""


def fuzzy_c_means(data, c, m, eps, max_iter):
    # data: input data
    # c: number of clusters
    # m: fuzziness parameter
    # eps: error tolerance
    # max_iter: maximum number of iterations

    # data
    N, n_features = data.shape  # N is the number of data points and n_features is the number of features in the dataset.

    # initialize u
    u = np.random.rand(N, c)  # Initializes the U-matrix (membership matrix) with random values between 0 and 1.
    u = u / np.sum(u, axis=1,
                   keepdims=True)  # Normalizes the U-matrix so that the sum of memberships for each data point is 1.

    # initialize v
    v = np.random.rand(c, n_features)  # Initializes the cluster centers (v) with random values.

    # iterative update
    for t in range(max_iter):  # Performs the fuzzy C-means algorithm for a maximum number of iterations.
        # save previous u and v
        u_prev = u.copy()  # Saves the current U-matrix before updating it.
        v_prev = v.copy()  # Saves the current cluster centers before updating them.

        # update u
        for i in range(c):  # For each cluster,
            for k in range(N):  # and for each data point,
                u[k, i] = 1.0 / np.sum(  # update the membership of the data point to the cluster using the FCM formula.
                    [(np.linalg.norm(data[k] - v[i]) / np.linalg.norm(data[k] - v[j])) ** (2 / (m - 1)) for j in
                     range(c)])

        # update v
        for i in range(c):  # For each cluster,
            v[i] = np.sum(u[:, i].reshape(N, 1) * data, axis=0) / np.sum(
                u[:, i])  # update the cluster center using the FCM formula.

        # check convergence
        if np.linalg.norm(
                v - v_prev) < eps:  # If the difference between the current and previous cluster centers is less than the error tolerance,
            break  # break the loop as the algorithm has converged.

        print(t)

    return u, v  # Returns the final U-matrix and cluster centers.


if __name__ == '__main__':
    # Read data from CSV
    data = read_csv_data('sensor_data.csv')

    # Apply Fuzzy C-Means clustering to calculating U-matrix and cluster centers
    data = data.values
    u_matrix, cntr = fuzzy_c_means(data, num_clusters, 2, 0.005, 1000)

    # Calculate MF parameters and plot MFs

    # Guassian MF
    # mf_params_gaussian = calculate_gaussian_mf_params(u_matrix, cntr, data)
    # plot_membership_functions(mf_params_gaussian, n_features, 'gaussian')

    # Triangular MF
    # mf_params_triangular = calculate_triangular_mf_params(u_matrix, cntr, data, 0.5)
    # plot_membership_functions(mf_params_triangular, n_features, 'triangular')

    # Trapezoidal MF
    mf_params_trapezoidal = calculate_trapezoidal_mf_params(u_matrix, cntr, data, 0.5, 0.5)
    plot_membership_functions(mf_params_trapezoidal, n_features, 'trapezoidal')