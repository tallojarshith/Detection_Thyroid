import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from thyroid_detection.exception import ThyroidException
from thyroid_detection.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.negative: int = 0
        self.compensated_hypothyroid: int = 1
        self.primary_hypothyroid: int = 2
        self.secondary_hypothyroid: int = 3
    
    def _asdict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
