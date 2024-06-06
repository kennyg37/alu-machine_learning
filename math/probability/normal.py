#!/usr/bin/env python3
"""The code bellow implements the normal probability distribution"""


class Normal:
    """Normal class that represents the normal probability
    distribution and calculates the probability
    of a given number of successes in a fixed interval
    of time or space.
    """
    π = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        self.data = data
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (
                sum([(x - self.mean) ** 2 for x in data]) / len(data)) ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value

        Args:
            x (int): the x-value

        Returns:
            float: the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score

        Args:
            z (int): the z-score

        Returns:
            float: the x-value of z
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given number of successes

        Args:
            x (int): number of successes

        Returns:
            float: the PDF value for x
        """

        return (Normal.e ** -((x - self.mean)**2 / (2 * self.stddev ** 2))) / (
            self.stddev * (2 * Normal.π) ** 0.5)

    def cdf(self, x):
        """Calculates the value of the CDF for a given number of successes

        Args:
            x (int): number of successes

        Returns:
            float: the CDF value for x
        """
        return (1 + Normal.erf((x - self.mean) / (self.stddev * 2 ** 0.5))) / 2
