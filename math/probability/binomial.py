#!/usr/bin/env python3
""" Binomial distribution mathematical operations"""


class Binomial:
    """ Binomial distribution class that represents a binomial distribution
    for a given number of trials and probability of success
    """

    def __init__(self, data=None, n=1, p=0.5):
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p = 1 - (variance / mean)
            n = round(mean / p)
            p = mean / n

            self.n = n
            self.p = float(p)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes

        Args:
            k (int): number of successes

        Returns:
            _type_: the PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i
        factorial_n = 1
        for i in range(1, self.n + 1):
            factorial_n *= i
        factorial_n_k = 1
        for i in range(1, self.n - k + 1):
            factorial_n_k *= i
        coefficient = factorial_n / (factorial_k * factorial_n_k)
        return coefficient * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes

        Args:
            k (int): number of successes

        Returns:
            _type_: the CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
