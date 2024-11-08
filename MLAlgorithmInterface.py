from abc import ABC, abstractmethod

class MLAlgorithm(ABC):
    """Abstract class for implementing the strategy pattern"""

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass