import numpy as np
import pandas as pd

from week_1.ex1.utils import compute_cost, gradient_descent

data = pd.read_csv('week_1/ex1/ex1data2.txt', sep=',', header=None)
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]

# Subtract the mean value of each feature from the dataset.
# After subtracting the mean, additionally scale (divide) the feature values
#  by their respective “standard deviations.”
X = (X - np.mean(X))/np.std(X)

# Added extra feature
X = np.hstack((np.ones((len(y), 1)), X))
alpha = 0.01
iterations = 400

theta = np.zeros((3, 1))
y = y[:, np.newaxis]

# Computing the cost
J = compute_cost(X, theta, y)
print(J)

# Finding the optimal parameters using Gradient Descent
theta = gradient_descent(X, y, theta, alpha, iterations)
print(theta)

J = compute_cost(X, theta, y)
print(J)
