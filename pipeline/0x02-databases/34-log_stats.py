#!/usr/bin/env python3
"""pymongo with nginx"""
from pymongo import MongoClient


if __name__ == "__main__":
    """provides some stats about Nginx logs stored in MongoDB"""
    client = MongoClient('mongodb://127.0.0.1:27017')
    doc_count = client.logs.nginx.count_documents({})
    method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    where = {"method": "GET", "path": "/status"}
    print('{} logs'.format(doc_count))
    print('Methods:')
    for i in method:
        count_method = doc_count.count_documents({"method": i})
        print('\tmethod {}: {}'.format(i, count_method))
    print("{} status check".format(doc_count.count_documents(where)))
