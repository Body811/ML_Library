from MLAlgorithmInterface import MLAlgorithm
import matplotlib.pyplot as plt
import numpy as np

class KMeans(MLAlgorithm):
    def __init__(self, k, max_iter, num_init):
        """
        Args:
            k (int): Number of clusters.
            max_iter (int): Maximum number of iterations.
            graph (bool): Whether to plot the cost function or not.
        """
        self.k = k
        self.max_iter = max_iter
        self.num_init = num_init

    def init_centroids(self, x):
        """
        This function initializes K centroids

        Args:
            x (ndarray): Data points

        Returns:
            centroids (ndarray): Initialized centroids
        """

        randx = np.random.permutation(x.shape[0])

        centroids = x[randx[: self.k]]

        return centroids

    def find_closest_centroids(self, x, centroids):
        """
        Computes the centroid memberships for every example

        Args:
            x (ndarray): (m, n) Input values
            centroids (ndarray): (K, n) centroids

        Returns:
            idx (tuple) : A tuple containing the cluster assignments
                         for each data point of shape (m,) and the cost.

        """

        distance = np.linalg.norm(x[:, np.newaxis] - centroids, axis=2)

        idx = np.argmin(distance, axis=1)

        cost = np.sum(np.min(distance, axis=1) ** 2) / x.shape[0]

        return idx, cost

    def compute_centroids(self, x, idx):
        """
        Returns the new centroids by computing the means of the
        data points assigned to each centroid.

        Args:
            x (ndarray):   (m, n) Data points
            idx (ndarray): (m,) Array containing index of closest centroid for each
                            example in x. Concretely, idx[i] contains the index of
                            the centroid closest to example i
            K (int):       number of centroids

        Returns:
            centroids (ndarray): (K, n) New centroids computed
        """

        n = x.shape[1]

        centroids = np.zeros((self.k, n))

        for i in range(self.k):
            centroids[i] = np.mean(x[idx == i], axis=0)

        return centroids

    def single_fit(self, x):
        centroids = self.init_centroids(x)
        for i in range(self.max_iter):
            idx, cost = self.find_closest_centroids(x, centroids)
            centroids = self.compute_centroids(x, idx)

        return centroids, idx, cost

    def fit(self, x):
        """
        Fit the k-means model to the data.

        Args:
            x (array): The input data of shape (m, n).

        Returns:
            None
        """
        best_centroids = None
        best_idx = None
        best_cost = float("inf")

        for i in range(self.num_init):
            centroids, idx, cost = self.single_fit(x)

            if cost < best_cost:
                best_centroids = centroids
                best_idx = idx
                best_cost = cost

        self.centroids = best_centroids
        self.idx = best_idx

    def predict(self, x):
        """
        Assign new data points to clusters based on the trained model.

        Args:
            x (array): The input data of shape (m, n).

        Returns:
            array: The cluster assignments for each data point of shape (m,).
        """
        distances = np.linalg.norm(x[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)