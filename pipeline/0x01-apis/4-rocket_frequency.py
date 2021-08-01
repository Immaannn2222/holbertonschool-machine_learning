#!/usr/bin/env python3
"""
API-REST
"""
import requests


if __name__ == '__main__':

    response = requests.get('https://api.spacexdata.com/v4/launches').json()
    scheduled_launches = {}
    for launch in response:
        rocket_id = launch['rocket']
        data = requests.get(
            'https://api.spacexdata.com/v4/rockets/' + rocket_id).json()
        rocket_name = data['name']
        if rocket_name not in scheduled_launches:
            scheduled_launches[rocket_name] = 1
        else:
            scheduled_launches[rocket_name] += 1

    for k, v in sorted(scheduled_launches.items(
    ), key=lambda k: k[1], reverse=True):
        print("{}: {}".format(k, v))
