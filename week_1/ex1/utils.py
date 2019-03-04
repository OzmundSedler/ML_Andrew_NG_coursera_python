import numpy as np


def predict_value(features: np.ndarray,
                  weights: np.ndarray,
                  ) -> np.ndarray:
    """
    Predict y_value for given features and weights vectors
    :param features: features vector
    :param weights: weights vector
    :return: predicted y value
    """
    return np.dot(features, weights)


def compute_cost(features: np.ndarray,
                 weights: np.ndarray,
                 y_fact: np.ndarray,
                 ) -> np.ndarray:
    """
    Compute cost function
    :param features: features vector
    :param weights: weights vector
    :param y_fact: vector of fact y values
    :return:
    """
    temp = predict_value(features, weights) - y_fact
    return np.sum(np.power(temp, 2)) / (2*len(features))


def gradient_descent(features: np.ndarray,
                     y_fact: np.ndarray,
                     weights: np.ndarray,
                     alpha: float,
                     iterations: int,
                     ) -> np.ndarray:
    """
    Finding the optimal parameters using Gradient Descent
    :param features: features vector
    :param weights: weights vector
    :param y_fact: vector of fact y values
    :param alpha: learning rate
    :param iterations: number of iterations
    :return:
    """
    for _ in range(iterations):
        temp = predict_value(features, weights) - y_fact
        temp = np.dot(features.T, temp)
        weights = weights - (alpha/len(y_fact))*temp
    return weights
