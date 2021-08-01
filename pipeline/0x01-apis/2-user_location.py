#!/usr/bin/env python3
"""
prints the location of a specific user/ GitHub api
"""
import requests
import sys
import datetime


if __name__ == '__main__':
    url = sys.argv[1]
    response = requests.get(url).json()
    if response.status_code == 404:
        print('Not found')
    elif response.status_code == 200:
        print(response['location'])
    elif response.status_code == 403:
        header = response.headers['X-Ratelimit-Reset']
        now = datetime.now().timestamp()
        time = (int(header) - int(now)) / 60
        print("Reset in {} min".format(time))
