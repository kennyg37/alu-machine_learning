#!/usr/bin/env python3
""" The code bellow implements the poisson probability distribution mathematically
it uses classes and more object oriented programming techniques to implement the
poisson probability distribution. 
"""

class Poisson:
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
            self.lambtha = (1/(sum(data) / len(data)))
            
            
        