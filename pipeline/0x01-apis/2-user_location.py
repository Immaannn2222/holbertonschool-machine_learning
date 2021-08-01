#!/usr/bin/env python3
"""
prints the location of a specific user/ GitHub api
"""
import requests
import sys
import datetime


if __name__ == '__main__':
    url = sys.argv[1]
    response = requests.get(url)
    if response.status_code == 200:
        print(response.json()['location'])
    elif response.status_code == 404:
        print('Not found')
    elif response.status_code == 403:
        chrono = datetime.fromtimestamp(
            int(response.headers['X-RateLimit-Reset'])).minute
        now = datetime.now().minute
        print("Reset in {} min".format(chrono - now))
