#!/usr/bin/env python3
""" This code calculates the correlation of a data set
    using numpy. The code includes a function correlation(C)
    that calculates the correlation of a data set.
"""
import numpy as np


def correlation(C):
    """ This function calculates the correlation of a data set.

    Args:
        C (numpy.ndarray): A numpy.ndarray of shape (d, d)
        containing a covariance matrix.
    """
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        raise TypeError("C must be a numpy.ndarray")
    d, d2 = C.shape
    if d != d2:
        raise ValueError("C must be a 2D square matrix")
    if np.any(C.T != C):
        raise ValueError("C must be a symmetric matrix")

    D = np.diag(1 / np.sqrt(np.diag(C)))
    corr = np.dot(np.dot(D, C), D)

    return corr
