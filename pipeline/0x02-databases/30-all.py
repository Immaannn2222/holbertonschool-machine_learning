#!/usr/bin/env python
"""pymongo"""


def list_all(mongo_collection):
    """ function that lists all documents in a collection """
    mongo_collection.find()
