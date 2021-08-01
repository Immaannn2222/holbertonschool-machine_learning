#!/usr/bin/env python3
"""
response-COLLECTION
"""
import requests


def availableShips(passengerCount):
    """ returns the list of ships that can hold a given number of passengers"""
    available = []
    SWAPI_url = 'https://swapi-api.hbtn.io/api/starships/?'
    response = requests.get(SWAPI_url).json()
    while response['next'] != None: ##handle the pagination
        for ship in response['results']:
            ship["passengers"].replace(',', '')
            if ship["passengers"].isdigit() and int(ship["passengers"]) >= passengerCount:
                available.append(ship['name'])
        SWAPI_url = response['next']
        response = requests.get(SWAPI_url).json()
    return available
