import os
import sys
import numpy as np
import pandas as pd
from thyroid_detection.entity.config_entity import ThyroidPredictorConfig
from thyroid_detection.entity.s3_estimator import thyroidEstimator
from thyroid_detection.exception import ThyroidException
from thyroid_detection.logger import logging
from thyroid_detection.utils.main_utils import read_yaml_file
from pandas import DataFrame


class ThyroidData:
    def __init__(self, sex: str, on_thyroxine: str, query_on_thyroxine: str,
                       on_antithyroid_medication: str, sick: str, pregnant: str, I131_treatment: str,
                       tumor: str, hypopituitary: str, psych: str,
                       age: int, TSH: float, T3: float, TT4: float, T4U: float, FTI: float):
        """
        Thyroid Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.age = age
            self.sex = sex
            self.on_thyroxine = on_thyroxine
            self.query_on_thyroxine = query_on_thyroxine
            self.on_antithyroid_medication = on_antithyroid_medication
            self.sick = sick
            self.pregnant = pregnant
            self.I131_treatment = I131_treatment
            self.tumor = tumor
            self.hypopituitary = hypopituitary
            self.psych = psych
            self.TSH = TSH
            self.T3 = T3
            self.TT4 = TT4
            self.T4U = T4U
            self.FTI = FTI

        except Exception as e:
            raise ThyroidException(e, sys) from e

    def get_thyroid_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from ThyroidData class input
        """
        try:
            thyroid_input_dict = self.get_thyroid_data_as_dict()
            return DataFrame(thyroid_input_dict)

        except Exception as e:
            raise ThyroidException(e, sys)

    def get_thyroid_data_as_dict(self):
        """
        This function returns a dictionary from ThyroidData class input
        """
        logging.info("Entered get_thyroid_data_as_dict method as ThyroidData class")

        try:
            input_data = {
                "age": [self.age],
                "sex": [self.sex],
                "on_thyroxine": [self.on_thyroxine],
                "query_on_thyroxine": [self.query_on_thyroxine],
                "on_antithyroid_medication": [self.on_antithyroid_medication],
                "sick": [self.sick],
                "pregnant": [self.pregnant],
                "I131_treatment": [self.I131_treatment],
                "tumor": [self.tumor],
                "hypopituitary": [self.hypopituitary],
                "psych": [self.psych],
                "TSH": [self.TSH],
                "T3": [self.T3],
                "TT4": [self.TT4],
                "T4U": [self.T4U],
                "FTI": [self.FTI],
            }

            logging.info("Created thyroid data dict")
            logging.info("Exited get_thyroid_data_as_dict method as ThyroidData class")

            return input_data

        except Exception as e:
            raise ThyroidException(e, sys)

class ThyroidClassifier:
    def __init__(self, prediction_pipeline_config: ThyroidPredictorConfig = ThyroidPredictorConfig()) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise ThyroidException(e, sys)

    def predict(self, dataframe) -> list:
        """
        This is the method of ThyroidClassifier
        Returns: Prediction in list format
        """
        try:
            logging.info("Entered predict method of ThyroidClassifier class")
            model = thyroidEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)

            if isinstance(result, np.ndarray):
                result = result.tolist()  # Convert NumPy array to list

            # Map numerical predictions to their respective classes
            target_mapping = {
                0: 'negative',
                1: 'compensated_hypothyroid',
                2: 'primary_hypothyroid',
                3: 'secondary_hypothyroid'
            }

            mapped_result = [target_mapping.get(pred, "Unknown") for pred in result]

            return mapped_result

        except Exception as e:
            raise ThyroidException(e, sys)
