#!/usr/bin/env python3
"""
DATA-COLLECTION
"""
import requests


def sentientPlanets():
    """returns the list of names of the home planets of all sentient"""
    SWAPI_url = "https://swapi-api.hbtn.io/api/species/?"
    response = requests.get(SWAPI_url).json()
    planets = []
    while response['next']:
        for species in response['results']:
                    if species['homeworld']:
                        planet = requests.get(species['homeworld']).json()
                        planets.append(planet['name'])
        SWAPI_url = response['next']
        response = requests.get(SWAPI_url).json()
    return planets
