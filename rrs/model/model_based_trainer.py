from ..data.data_reader import DataReader
from .trainer.model_trainer import ModelTrainer
import abc


class ModelBasedTrainer(object):

    def __init__(self, data_reader: DataReader, model_trainer: ModelTrainer, two_class: bool = False):
        self.data_reader = data_reader
        self.model_trainer = model_trainer
        self.two_class = two_class

    @abc.abstractmethod
    def train_models(self):
        pass
