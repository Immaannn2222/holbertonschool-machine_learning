#!/usr/bin/env python3
"""
DATA-COLLECTION
"""
import requests


def sentientPlanets():
    """returns the list of names of the home planets of all sentient"""
    response = requests.get("https://swapi-api.hbtn.io/api/species/").json()
    planets = []
    while True:
        results = response['results']
        for sentinent in results:
            if sentinent['homeworld']:
                name = requests.get(sentinent['homeworld']).json()['name']
                planets.append(name)
        if response['next'] is None:
            break
        response = requests.get(response['next'])
        response = response.json()
    return planets
