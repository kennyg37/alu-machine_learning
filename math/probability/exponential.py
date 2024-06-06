#!/usr/bin/env python3
""" The code bellow implements the exponential probability distribution
mathematically it uses classes and more object oriented
"""


class Exponential:
    """ Exponential class that represents the exponential probability
    distribution and calculates the probability
    of a given number of successes in a fixed interval
    of time or space.
    """
    π = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = lambtha
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """Calculates the value of the PDF for a given number of successes

        Args:
            x (int): number of successes

        Returns:
            float: the PDF value for x
        """

        if x < 0:
            return 0
        return self.lambtha * (Exponential.e ** (-self.lambtha * x))

    def cdf(self, x):
        """Calculates the value of the CDF for a given number of successes

        Args:
            x (int): number of successes

        Returns:
            float: the CDF value for x
        """
        if x < 0:
            return 0
        return 1 - Exponential.e ** (-self.lambtha * x)
