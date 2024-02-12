import time

import numpy as np
from sklearn.metrics import silhouette_score

class AbstractFCM:

    # Constructor for the AbstractFCM class.
    # @param data: input data
    # @param c: number of clusters
    # @param m: fuzziness parameter
    # @param eps: error tolerance
    # @param max_iter: maximum number of iterations
    #
    def __init__(self, data, c, m, eps, max_iter):
        self._c = c
        self._m = m
        self._eps = eps
        self._max_iter = max_iter
        self._data = data

        # N is the number of data points and n_features is the number of features in the dataset.
        self._n, self._n_features = data.shape

        # Initializes the U-matrix (membership matrix) with random values between 0 and 1.
        self._u = self._initialize_u()

        # Initializes the cluster centers (v) with random values.
        self._v = self._initialize_v()

    # Initializes the U-matrix (membership matrix) with random values between 0 and 1.
    def _initialize_u(self):
        raise NotImplementedError("Please Implement this method")

    # Initializes the cluster centers (v) with random values.
    def _initialize_v(self):
        raise NotImplementedError("Please Implement this method")

    # update the membership of the data point to the cluster using the FCM formula.
    def _update_u(self):
        raise NotImplementedError("Please Implement this method")

    # update the cluster center using the FCM formula.
    def _update_v(self):
        raise NotImplementedError("Please Implement this method")

    # Performs the fuzzy C-means algorithm.
    # @return: final U-matrix and cluster centers
    def fuzzy_c_means(self):

        # initialize u
        # self._initialize_u()

        # initialize v
        # self._initialize_v()

        print("Fuzzy C-means algorithm started...")
        start_time = time.time()  # Capture the start time

        # iterative update
        for t in range(self._max_iter):  # Performs the fuzzy C-means algorithm for a maximum number of iterations.

            # print("Iteration: ", t)
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"\rIteration: {t}, Elapsed Time: {elapsed_time:.2f} seconds", end="")
            # self.print_progress_bar(t + 1, self._max_iter, prefix='Progress:', suffix='Complete', length=50)

            # save previous u and v
            u_prev = self._u.copy()  # Saves the current U-matrix before updating it.
            v_prev = self._v.copy()  # Saves the current cluster centers before updating them.

            # update u
            self._update_u()

            # update v
            self._update_v()

            # check convergence
            if np.linalg.norm(
                    self._v - v_prev) < self._eps:  # If the difference between the current and previous cluster
                # centers is less than the error tolerance,
                break  # break the loop as the algorithm has converged.

        print("\nFuzzy C-means algorithm finished.\n")

        return self._u, self._v  # Returns the final U-matrix and cluster centers.

    def plot_cluster(self):
        # TODO: Implement this method
        raise NotImplementedError("Please Implement this method")

    def jaccard_coefficient(self):
        # TODO: Implement this method
        raise NotImplementedError("Please Implement this method")

    def silhouette_score(self):
        raise NotImplementedError("Please Implement this method")


class ConcreteFCM(AbstractFCM):

    def __init__(self, data, c, m, eps, max_iter):
        super().__init__(data, c, m, eps, max_iter)

    def _initialize_u(self):
        # Initializes the U-matrix (membership matrix) with random values between 0 and 1.
        u = np.random.rand(self._n, self._c)
        # Normalizes the U-matrix so that the sum of memberships for each data point is 1.
        u = u / np.sum(u, axis=1, keepdims=True)
        return u

    def _initialize_v(self):
        # Initializes the cluster centers (v) with random values.
        v = np.random.rand(self._c, self._n_features)
        return v

    def _update_u(self):
        for i in range(self._c):  # For each cluster,
            for k in range(self._n):  # and for each data point,
                self._u[k, i] = 1.0 / np.sum(
                    # update the membership of the data point to the cluster using the FCM formula.
                    [(np.linalg.norm(self._data[k] - self._v[i]) / np.linalg.norm(self._data[k] - self._v[j])) ** (
                            2 / (self._m - 1)) for j in
                     range(self._c)])

    def _update_v(self):
        for i in range(self._c):  # For each cluster,
            self._v[i] = np.sum(self._u[:, i].reshape(self._n, 1) * self._data, axis=0) / np.sum(
                self._u[:, i])  # update the cluster center using the FCM formula.

    def silhouette_score(self):
        # Convert fuzzy memberships to hard memberships by assigning each sample to the cluster with the highest
        # membership
        labels = np.argmax(self._u, axis=1)

        # Use sklearn's silhouette_score function
        score = silhouette_score(self._data, labels)

        return score
