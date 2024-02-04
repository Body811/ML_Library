import matplotlib.pyplot as plt
import numpy as np
import time
from abc import ABC, abstractmethod


class MLAlgorithm(ABC):
    """Abstract class for implementing the strategy pattern"""

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class LinearRegression(MLAlgorithm):
    def __init__(self, alpha, epsilon, theta, bias, max_iter, graph):
        """
        Args:
            alpha: Learning rate.
            epsilon: Threshold for determining convergence.
            theta: Initial guess for theta. If None, use the zero vector.
            bias: The bias (intercept) term.
            max_iter: max iterations before the fit function terminates.
            graph: variable for wheather you want to plot the cost function or not.
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.theta = theta
        self.bias = bias
        self.max_iter = max_iter
        self.graph = graph

    def _compute_cost(self, x, y, theta, bias, m):
        """
        computes the cost function for linear regression.

        Args:
            x (array): The input features of shape (m, n).
            y (array): The target values of shape (m,).
            theta (array): Initial guess for theta. If None, use the zero vector.
            bias (scaler): The bias (intercept) term.
            m (scalar) : Number of training examples

        Returns:
            cost (float): The cost of using theta, bias as the parameters for linear regression
                to fit the data points in x and y
        """

        h = np.dot(x, theta) + bias

        cost = np.sum((h - y) ** 2)
        cost = cost / (2 * m)

        return cost

    def compute_gradient(x, y, theta, bias, m):
        """
        computes the gradient for linear regression .

        Args:
            h (array): the predicted values of x shape(m,).
            x (array): The input features of shape (m, n).
            y (array): The target values of shape (m,).
            theta (array): Initial guess for theta. If None, use the zero vector.
            m (scalar) : Number of training examples

        Returns:
            gradient_theta (scalar): The gradient of the cost with respect to theta
            gradient_bias (scalar): The gradient of the cost with respect to bias
        """
        gradient_theta = np.zeros_like(theta)
        gradient_bias = 0

        h = np.dot(x, theta) + bias

        gradient_theta += np.dot((h - y), x) / m
        gradient_bias += np.sum(h - y) / m

        return gradient_theta, gradient_bias

    def fit(self, x, y):
        """
        Fit the linear regression model to the training data using gradient descent.

        Args:
            x (array): The input features of shape (m, n).
            y (array): The target values of shape (m,).

        Returns:
            None
        """

        m, n = x.shape

        if self.theta == None:
            self.theta = np.zeros(n)

        cost_values = []
        prev_cost = float("inf")

        for i in range(self.max_iter):
            gradient_theta, gradient_bias = self.compute_gradient(
                x, y, self.theta, self.bias, m
            )
            self.theta = self.theta - (self.alpha * gradient_theta)
            self.bias = self.bias - (self.alpha * gradient_bias)

            cost = self._compute_cost(x, y, self.theta, self.bias, m)
            cost_values.append(cost)

            if i % 10 == 0:
                print(f"Iteration {i:4d}: Cost {cost_values[-1]:8.2f}   ")

            if abs(cost - prev_cost) < self.epsilon:
                print(f"Converged after {i} iterations.")
                break

            prev_cost = cost

        if self.graph:
            plt.plot(100 + np.arange(len(cost_values[100:])), cost_values[100:])
            plt.ylabel("Cost")
            plt.xlabel("Iterations")
            plt.title("Cost vs. Iterations")
            plt.show()

    def predict(self, x):
        """
        Make predictions on new data using the trained linear regression model.

        Args:
            x (array): The input features of shape (m, n).

        Returns:
            array: The predicted target values of shape (m,).
        """

        return np.dot(x, self.theta) + self.bias


class LogisticRegression(MLAlgorithm):
    def __init__(self, alpha, epsilon, theta, bias, max_iter, graph):
        """
        Args:
            alpha: Learning rate.
            epsilon: Threshold for determining convergence.
            theta: Initial guess for theta. If None, use the zero vector.
            bias: The bias (intercept) term.
            max_iter: max iterations before the fit function terminates.
            graph: variable for wheather you want to plot the cost function or not.
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.theta = theta
        self.bias = bias
        self.max_iter = max_iter
        self.graph = graph

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, x, y, theta, bias, m):
        """
        computes the cost function for logistic regression.

        Args:
            x (array): The input features of shape (m, n).
            y (array): The target values of shape (m,).
            theta (array): Initial guess for theta. If None, use the zero vector.
            bias (scaler): The bias (intercept) term.
            m (scalar) : Number of training examples

        Returns:
            cost (float): The cost of using theta, bias as the parameters for logistic regression
                to fit the data points in x and y
        """
        h = self.sigmoid((np.dot(x, theta) + bias))

        pos_loss = -y * np.log(h)
        neg_loss = -(1 - y) * np.log(1 - h)
        total_loss = pos_loss + neg_loss

        cost = np.sum(total_loss) / m

        return cost

    def compute_gradient(x, y, theta, bias, m):
        """
        computes the gradient for logistic regression .

        Args:
            h (array): the predicted values of x shape(m,).
            x (array): The input features of shape (m, n).
            y (array): The target values of shape (m,).
            theta (array): Initial guess for theta. If None, use the zero vector.
            m (scalar) : Number of training examples

        Returns:
            gradient_theta (scalar): The gradient of the cost with respect to theta
            gradient_bias (scalar): The gradient of the cost with respect to bias
        """
        gradient_theta = np.zeros_like(theta)
        gradient_bias = 0

        h = np.dot(x, theta) + bias

        gradient_theta += np.dot((h - y), x) / m
        gradient_bias += np.sum(h - y) / m

        return gradient_theta, gradient_bias

    def fit(self, x, y):
        """
        Fit the linear regression model to the training data using gradient descent.

        Args:
            x (array): The input features of shape (m, n).
            y (array): The target values of shape (m,).

        Returns:
            None
        """

        m, n = x.shape

        if self.theta == None:
            self.theta = np.zeros(n)

        cost_values = []

        h = self.sigmoid((np.dot(x, self.theta) + self.bias))

        for i in range(self.max_iter):
            gradient_theta, gradient_bias = self.compute_gradient(
                h, x, y, self.theta, m
            )
            self.theta = self.theta - (self.alpha * gradient_theta)
            self.bias = self.bias - (self.alpha * gradient_bias)

            if i <= 100000:
                cost_values.append(self._compute_cost(x, y, self.theta, self.bias, m))

            if i % 10 == 0:
                print(f"Iteration {i:4d}: Cost {cost_values[-1]:8.2f}   ")

    def predict(self, x):
        """
        Make predictions on new data using the trained logistic regression model.

        Args:
            x (array): The input features of shape (m, n).

        Returns:
            array: The predicted target values of shape (m,).
        """

        h = self.sigmoid((np.dot(x, self.theta) + self.bias))

        p = [1 if x >= 0.5 else 0 for x in h]

        return np.array(p)


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


class MLModel:
    """Wrapper class used for implementing the strategy pattern"""

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def set_algorithm(self, algorithm):
        """
        Changes the active algorithm

        Args:
            algorithm (MLAlgorithm): The algorithm you want to change to

        Return:
            None
        """

        self.algorithm = algorithm

    def fit(self, x, y):
        return self.algorithm.fit(x, y)

    def predict(self, x):
        return self.algorithm.predict(x)


class MLAlgorithmLibrary:
    """
    Library for creating machine learning models.

    This class provides a static method to create a linear regression model.

    Usage Example:
        model = MLAlgorithmLibrary.create_linear_regression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """

    @staticmethod
    def create_linear_regression(
        alpha=0.00001, epsilon=0.001, theta=None, bias=0, max_iter=1000, graph=False
    ):
        """
        Create a linear regression model.

        Returns:
            MLModel: A linear regression model.
        """
        return MLModel(LinearRegression(alpha, epsilon, theta, bias, max_iter, graph))

    @staticmethod
    def create_logistic_regression(
        alpha=0.00001, epsilon=0.001, theta=None, bias=0, max_iter=1000, graph=False
    ):
        """
        Create a logistic regression model.

        Returns:
            MLModel: A logistic regression model.
        """
        return MLModel(LogisticRegression(alpha, epsilon, theta, bias, max_iter, graph))

    @staticmethod
    def create_kmeans(k=2, max_iter=1000, num_init=1):
        """
        Create a Kmean unsupervised learning model.

        Returns:
            MLModel: A kmeans model.
        """
        return MLModel(KMeans(k, max_iter, num_init))

    @staticmethod
    def compute_accuracy(prediction, y):
        return np.mean(prediction == y) * 100
