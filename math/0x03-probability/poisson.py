#!/usr/bin/env python3
""" task 0 """


class Poisson:
    """ represents a poisson distribution """
    def __init__(self, data=None, lambtha=1.):
        """ class constructor """
        self.data = data
        if data is None:
            if (lambtha <= 0):
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if isinstance(data, list) is False:
                raise TypeError("data must be a list")
            if(len(data)) < 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = (sum(data) / len(data))
