#!/usr/bin/env python3
"""pymongo"""


def update_topics(mongo_collection, name, topics):
    """changes all topics of a school document based on the name"""
    x = mongo_collection.update_many(
        {'name': name}, {'$set': {'topics': topics}})
    return x
