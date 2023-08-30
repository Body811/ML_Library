import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod


class MLAlgorithm(ABC):
    """ Abstract class for implementing the strategy pattern """
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class linearRegression(MLAlgorithm):
     
    
    def __init__(self, alpha=0.02,epsilon=0.0001, theta=None):
        """
        Args:
            alpha: Learning rate  
            epsilon: Threshold for determining convergence.
            theta: Initial guess for theta. If None, use the zero vector.
        """
        self.theta = theta
        self.alpha = alpha
        self.epsilon = epsilon

    def fit(self, x, y):   
        """
        Fit the linear regression model to the training data using gradient descent.

        Args:
            x (array): The input features of shape (m, n).
            y (array): The target values of shape (m,).

        Returns:
            None
        """
        
        converged = False
        m, n = x.shape     
        
        if self.theta == None:
            self.theta = np.zeros(n)
            
        while not converged:
            y_pred = np.dot(x, self.theta)

            error = y - y_pred

            gradiant = np.dot(x.T, error) / m
            
            self.theta += self.alpha * gradiant
            
            
            if np.linalg.norm(gradiant) < self.epsilon:
                converged = True

    def predict(self, x):
        """
        Make predictions on new data using the trained linear regression model.

        Args:
            x (array): The input features of shape (m, n).

        Returns:
            array: The predicted target values of shape (m,).
        """
        
        return np.dot(x, self.theta)
        


class MLModel(MLAlgorithm):
    """Wrapper class used for implementing the strategy pattern"""

    
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def set_algorithm(self,algorithm):
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
    def create_linear_regression():
        """
        Create a linear regression model.

        Returns:
            MLModel: A linear regression model.
        """
        return MLModel(linearRegression())
