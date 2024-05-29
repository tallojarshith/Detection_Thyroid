import sys
from thyroid_detection.exception import ThyroidException
from thyroid_detection.logger import logging
from thyroid_detection.components.data_ingestion import DataIngestion
from thyroid_detection.entity.config_entity import DataIngestionConfig
from thyroid_detection.entity.artifact_entity import DataIngestionArtifact

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from MongoDB")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from MongoDB")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise ThyroidException(e, sys) from e

    def run_pipeline(self,)->None:
        try:
            data_ingestion_artifact=self.start_data_ingestion()
        except Exception as e:
            raise ThyroidException(e, sys) from e
