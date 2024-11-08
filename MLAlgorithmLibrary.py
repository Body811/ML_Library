from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
from KMeans import KMeans
import numpy as np






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
        return LinearRegression(alpha, epsilon, theta, bias, max_iter, graph)

    @staticmethod
    def create_logistic_regression(
        alpha=0.00001, epsilon=0.001, theta=None, bias=0, max_iter=1000, graph=False
    ):
        """
        Create a logistic regression model.

        Returns:
            MLModel: A logistic regression model.
        """
        return LogisticRegression(alpha, epsilon, theta, bias, max_iter, graph)

    @staticmethod
    def create_kmeans(k=2, max_iter=1000, num_init=1):
        """
        Create a Kmean unsupervised learning model.

        Returns:
            MLModel: A kmeans model.
        """
        return KMeans(k, max_iter, num_init)

    @staticmethod
    def compute_accuracy(prediction, y):
        return np.mean(prediction == y) * 100
