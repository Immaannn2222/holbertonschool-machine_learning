#!/usr/bin/env python3
"""pymongo"""
from pymongo import MongoClient


if __name__ == "__main__":
    """provides some stats about Nginx logs stored in MongoDB"""
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_collection = client.logs.nginx
    doc_count = logs_collection.count_documents({})

    method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    search = {"method": "GET", "path": "/status"}
    print('{} logs'.format(doc_count))
    print('Methods:')
    for i in method:
        count_method = logs_collection.count_documents({"method": i})
        print('\tmethod {}: {}'.format(i, count_method))
    print("{} status check".format(logs_collection.count_documents(search)))
