#!/usr/bin/env python3
"""
prints the location of a specific user/ GitHub api
"""
import requests
import sys
import datetime


if __name__ == '__main__':
    """ script that prints the location of a specific user"""
    url = sys.argv[1]
    response = requests.get(url)
    metadata = response.json()
    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 200:
        print(metadata["location"])
    elif response.status_code == 403:
        limit = response.headers["X-Ratelimit-Reset"]
        timer = (int(limit) - int(datetime.now().timestamp())) / 60
        print("Reset in {} min".format(int(timer)))
