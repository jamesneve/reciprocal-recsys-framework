from .data.data_reader import DataReader
from .trainer.model_trainer import ModelTrainer


class ModelBasedTrainer(object):

    def __init__(self, data_reader: DataReader, model_trainer: ModelTrainer):
        self.data_reader = data_reader
        self.model_trainer = model_trainer

    def train_model(self):
        users_df = self.data_reader.read_data()
        model = self.model_trainer.train(users_df)
        return users_df
