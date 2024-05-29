import os,sys
import pymongo
import certifi
from thyroid_detection.constants import DATABASE_NAME, MONGODB_URL_KEY
from thyroid_detection.exception import ThyroidException
from thyroid_detection.logger import logging

ca = certifi.where()

class MongoDBClient:
    """
    Class Name :   MongoDBClient
    Description :   This class establishes a connection to the MongoDB database and provides
                    methods to interact with the database.
    
    Output      :   connection to MongoDB database
    On Failure  :   raises an exception
    """
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection successful")
        except Exception as e:
            raise ThyroidException(e, sys)
