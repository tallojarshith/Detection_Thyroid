import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from thyroid_detection.entity.config_entity import DataIngestionConfig
from thyroid_detection.entity.artifact_entity import DataIngestionArtifact
from thyroid_detection.exception import ThyroidException
from thyroid_detection.logger import logging
from thyroid_detection.data_access.thyroid_data import ThyroidData

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ThyroidException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name: export_data_into_feature_store
        Description: This method exports data from MongoDB to a CSV file
        
        Output: Data is returned as an artifact of data ingestion components
        On Failure: Write an exception log and then raise an exception
        """
        try:
            logging.info("Exporting data from MongoDB")
            thyroid_data = ThyroidData()
            dataframe = thyroid_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise ThyroidException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Method Name: split_data_as_train_test
        Description: This method splits the dataframe into train set and test set based on the split ratio 
        
        Output: Folder is created in the file system
        On Failure: Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of DataIngestion class")

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited split_data_as_train_test method of DataIngestion class")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info("Exporting train and test file paths.")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Exported train and test file paths.")
        except Exception as e:
            raise ThyroidException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name: initiate_data_ingestion
        Description: This method initiates the data ingestion components of the training pipeline 
        
        Output: Train set and test set are returned as the artifacts of data ingestion components
        On Failure: Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from MongoDB")
            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")
            logging.info("Exited initiate_data_ingestion method of DataIngestion class")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise ThyroidException(e, sys) from e
