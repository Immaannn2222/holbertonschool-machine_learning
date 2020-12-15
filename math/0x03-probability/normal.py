#!/usr/bin/env python3
""" 3rd dis """


class Normal:
    """the normal class"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """class constructor"""
        self.data = data
        self.mean = float(mean)
        self.stddev = float(stddev)
        y = 0
        if data is None:
            if (stddev <= 0):
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
        else:
            if isinstance(data, list) is False:
                raise TypeError("data must be a list")
            if(len(data)) < 3:
                raise ValueError("data must contain multiple values")
            self.mean = (sum(data) / len(data))
            for i in data:
                y += (i - self.mean) ** 2
            self.stddev = (y / len(data)) ** (1/2)

    def z_score(self, x):
        """ Calculates the z-score of a given x-value """
        return((x - self.mean) / self.stddev)

    def x_value(self, z):
        """ Calculates the x-value of a given z-score """
        return(self.mean + z * self.stddev)

    def err_f(self, x):
        """ clacultes the error in a function """
        π = 3.1415926536
        deno = 2 / π ** 0.5
        up = x - x ** 3 / 3 + x ** 5 / 10 - x ** 7 / 42 + x ** 9 / 216
        return(up / deno)

    def pdf(self, x):
        """ Calculates the value of the PDF for a given x-value """
        π = 3.1415926536
        e = 2.7182818285
        i = e ** (- (1 / 2) * (((x - self.mean) / self.stddev) ** 2))
        j = (self.stddev * ((2 * π) ** (1/2)))
        return(i / j)

    def cdf(self, x):
        """ Calculates the value of the CDF for a given x-value """
        π = 3.1415926536
        e = 2.7182818285
        y = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return(0.5 * (1 + self.err_f(y)))
