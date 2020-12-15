#!/usr/bin/env python3
""" expo """


class Exponential:
    """class"""
    def __init__(self, data=None, lambtha=1.):
        """ class constructor """
        self.data = data
        self.lambtha = float(lambtha)
        if data is None:
            if (lambtha <= 0):
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if isinstance(data, list) is False:
                raise TypeError("data must be a list")
            if(len(data)) < 3:
                raise ValueError("data must contain multiple values")
            self.lambtha = (len(data) / sum(data))

    def pdf(self, x):
        """ Calculates the value of the PMF time period """
        e = 2.7182818285
        if x < 0:
            return 0
        return(e ** (-self.lambtha * x) * self.lambtha)

    def cdf(self, x):
        """  value of the PDF for a given time period """
        e = 2.7182818285
        if x < 0:
            return(0)
        return(1 - (e ** (-self.lambtha * x)))
