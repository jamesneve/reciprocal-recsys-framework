from .data.data_reader import DataReader
from .trainer.model_trainer import ModelTrainer
import numpy as np


class ModelBasedTrainer(object):

    def __init__(self, data_reader: DataReader, model_trainer: ModelTrainer):
        self.data_reader = data_reader
        self.model_trainer = model_trainer

    def train_model(self, model_dir):
        pass
