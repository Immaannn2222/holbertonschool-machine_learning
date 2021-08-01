#!/usr/bin/env python3
"""
prints the location of a specific user/ GitHub api
"""
import sys
import requests
import time

if __name__ == "__main__":
    """
    If the status code is 403, print Reset in X min
    where X is the number of minutes from now and
    the value of X-Ratelimit-Reset
    """
    user = sys.argv[1]
    response = requests.get(user)
    corpus = response.json()
    if response .status_code == 404:
        print("Not found")
    elif response .status_code == 200:
        print(corpus["location"])
    elif response .status_code == 403:
        limit = response.headers["X-Ratelimit-Reset"]
        x = (int(limit) - int(time.time())) / 60
        print("Reset in {} min".format(int(x)))
