from ML_algorithms import MLAlgorithmLibrary

import numpy as np

linear_regression = MLAlgorithmLibrary.create_linear_regression()


# Create a test dataset
X_train = np.array([[1, 1], 
                    [2, 2], 
                    [3, 3], 
                    [4, 4]])  # Input features

y_train = np.array([2, 4, 6, 8])  # Target values

# Create a linear regression model
model = MLAlgorithmLibrary.create_linear_regression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Create a test dataset
X_test = np.array([[5, 5],
                   [6, 6]])  #input features

# Make predictions on the test data
predictions = model.predict(X_test)

# Print the predictions
print("Predictions:", predictions)
