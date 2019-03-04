import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from week_1.ex1.utils import compute_cost, gradient_descent

data = pd.read_csv('week_1/ex1/ex1data1.txt', header=None)
X = data.ix[:, 0]
y = data.ix[:, 1]

plt.scatter(X, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

X = X[:, np.newaxis]
y = y[:, np.newaxis]

# Added extra feature
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Initialize weight vector
theta = np.zeros([2, 1])

# Computing the cost
J = compute_cost(X, theta, y)
print(J)

# Finding the optimal parameters using Gradient Descent
iterations = 1500
alpha = 0.01
theta = gradient_descent(X, y, theta, alpha, iterations)
print(theta)

# We now have the optimized value of weights.
# Use this value in the above cost function.
J = compute_cost(X, theta, y)
print(J)

# Plot the result
plt.scatter(X[:, 1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:, 1], np.dot(X, theta))
plt.show()
