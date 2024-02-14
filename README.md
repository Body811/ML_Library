# MLAlgorithmLibrary

MLAlgorithmLibrary is a scikit learn like Python library, encompassing a diverse array of machine learning algorithms. 

## Installation

```
git clone https://github.com/Body811/ML_Library.git
```
## Features
<ul>
<li>Create linear regression models</li>
<li>Create logistic regression models</li>
<li>Create KMeans clustering models</li>
<li>Compute accuracy of predictions</li>
</ul>

## Usage
### Import
```
from ML_algorithms import MLAlgorithmLibrary
```
### Creating a linear regression model
```
model = MLAlgorithmLibrary.create_linear_regression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
### Creating logistics regression model
```
model = MLAlgorithmLibrary.create_logistic_regression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
### Creating Kmeans clustring model
```
model = MLAlgorithmLibrary.create_kmeans()
model.fit(X)
clusters = model.predict(X)
```
### Compute accuracy
```
accuracy = MLAlgorithmLibrary.compute_accuracy(predictions, y_test)
print("Accuracy:", accuracy)
```

