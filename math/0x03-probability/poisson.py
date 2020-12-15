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
            if(len(data)) < 3:
                raise ValueError("data must contain multiple values")
            self.lambtha = (sum(data) / len(data))

    def pmf(self, k):
        """ Calculates the value of the PMF of “successes”"""
        e = 2.7182818285
        fact = 1
        if isinstance(k, int) is False:
            int(k)
        for i in range(2, k + 1):
            fact *= i
        return((e ** (-self.lambtha) * self.lambtha ** k) / fact)
