from thyroid_detection.entity import artifact_entity, config_entity
from thyroid_detection.exception import ThyroidException
from thyroid_detection.logger import logging
from typing import Optional
import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from thyroid_detection.utils import main_utils
from thyroid_detection.constants import TARGET_COLUMN


class DataTransformation:
    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>' * 20} Data Transformation {'<<' * 20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise ThyroidException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            Categorical_Features = [
                "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick",
                "pregnant", "I131_treatment", "tumor", "hypopituitary", "psych"
            ]
            Numerical_Features = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
            
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
                ]
            )
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", missing_values=np.nan)),
                    ("robust_scaler", RobustScaler())
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ("num", numeric_transformer, Numerical_Features),
                    ("cat", categorical_transformer, Categorical_Features)
                ]
            )
            
            pipeline = Pipeline([("preprocessor", preprocessor)])
            return pipeline
        except Exception as e:
            raise ThyroidException(e, sys)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            # Reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            # Selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            # Transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            # Transforming input features
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            random_over_sampler = RandomOverSampler(random_state=42)
            logging.info(
                f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}"
            )
            input_feature_train_arr, target_feature_train_arr = random_over_sampler.fit_resample(
                input_feature_train_arr, target_feature_train_arr
            )
            logging.info(
                f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}"
            )

            logging.info(
                f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}"
            )
            input_feature_test_arr, target_feature_test_arr = random_over_sampler.fit_resample(
                input_feature_test_arr, target_feature_test_arr
            )
            logging.info(
                f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}"
            )

            # Combining input features and target features
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # Save numpy array
            logging.info("Saving transformed training and testing arrays")
            main_utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path, array=train_arr
            )
            main_utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path, array=test_arr
            )

            # Save transformation pipeline and label encoder
            logging.info("Saving transformation pipeline and label encoder")
            main_utils.save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=transformation_pipeline,
            )
            

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ThyroidException(e, sys)
