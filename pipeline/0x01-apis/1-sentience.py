#!/usr/bin/env python3
"""
DATA-COLLECTION
"""
import requests


def sentientPlanets():
    """returns the list of names of the home planets of all sentient"""
    response = requests.get("https://swapi-api.hbtn.io/api/species/").json()
    planets = []
    while response['next']:
        for sentinent in response['results']:
            specie = sentinent['homeworld']
            if specie:
                sent = requests.get(specie).json()['name']
                planets.append(sent)
        response = requests.get(response['next']).json()
    return planets
