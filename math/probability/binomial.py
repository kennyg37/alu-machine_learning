#!/usr/bin/env python3
""" Binomial distribution mathematical operations"""


class Binomial:
    """ Binomial distribution class that represents a binomial distribution 
    for a given number of trials and probability of success
    """

    def __init__(self, data=None, n=1, p=0.5):
        self.data = data
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n < 0:
                raise ValueError("n must be a positive integer")
            if p <= 0 or p > 1:
                raise ValueError("p must be grater than 0 and less than 1")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            p = sum(data) / len(data)
