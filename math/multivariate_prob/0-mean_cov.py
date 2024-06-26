#!/usr/bin/env python3
""" This code calculates the mean and covariance of a data set
    using numpy. The code includes a function mean_cov(X) that
    calculates the mean and covariance of a data set.
"""
import numpy as np


def mean_cov(X):
    """ This function calculates the mean and covariance of a data set.

    Args:
        X (numpy.ndarray): A numpy.ndarray of shape (n, d)
        where n is the number
        of data points and d is the number of dimensions.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_c = X - mean
    cov = np.dot(X_c.T, X_c) / (n - 1)

    return mean, cov
