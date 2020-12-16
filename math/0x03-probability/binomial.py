#!/usr/bin/env python3
""" 4th dis """


class Binomial:
    """ binomial class """
    def __init__(self, data=None, n=1, p=0.5):
        """ class constructor """
        self.data = data
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            if (p < 0) is True or (p > 1) is True:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if isinstance(data, list) is False:
                raise TypeError("data must be a list")
            if(len(data)) < 3:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            y = 0
            for x in data:
                y += float((x - mean) ** 2)
            v = float(y / (len(data)))
            self.p = 1 - v / mean
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def fact(self, k):
        """ calculates fact"""
        f = 1
        for i in range(2, k + 1):
            f *= i
        return f

    def pmf(self, k):
        """ Calculates the value of the PMF of “successes”"""
        if isinstance(k, int) is False:
            k = int(k)
        if k < 0:
            return 0
        a = self.fact(self.n) / (self.fact(k) * self.fact(self.n - k))
        b = a * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return (b)

    def cdf(self, k):
        """ Calculates the value of the CDF of “successes” """
        pmff = 0
        if isinstance(k, int) is False:
            k = int(k)
        if k < 0:
            return 0
        for i in range(k + 1):
            pmff += self.pmf(i)
        return (pmff)
