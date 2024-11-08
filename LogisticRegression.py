from MLAlgorithmInterface import MLAlgorithm
import matplotlib.pyplot as plt
import numpy as np

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
