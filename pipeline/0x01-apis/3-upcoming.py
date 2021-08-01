#!/usr/bin/env python3
"""
API python
"""
import requests


if __name__ == '__main__':
    address = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(address).json()
    information = {
        'launch name': '',
        'date': '',
        'rocket id': '',
        "date_local": '',
        'launchpad id': ''}
    for line in response:
        date = line['date_unix']
        if information['date'] == '' or information['date'] > date:
            information['date'] = date
            information['launch name'] = line['name']
            information['rocket id'] = line['rocket']
            information['launchpad id'] = line['launchpad']
            information['date_local'] = line['date_local']
    the_rocket = requests.get(
        'https://api.spacexdata.com/v4/rockets/' +
        information['rocket id']).json()
    rocket_name = the_rocket['name']

    launch = requests.get(
        'https://api.spacexdata.com/v4/launchpads/' +
        information['launchpad id']).json()
    launchpad_name = launch['name']
    launchpad_locality = launch['locality']

    date = information['date_local']
    name = information['launch name']
    print(
        "{} ({}) {} - {} ({})".format(
            name,
            date,
            rocket_name,
            launchpad_name,
            launchpad_locality))
