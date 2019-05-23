from .data.data_reader import DataReader
from .trainer.model_trainer import ModelTrainer
from .model_based_trainer import ModelBasedTrainer
import numpy as np


class LatentFactorTrainer(ModelBasedTrainer):

    USER_FILENAME = "user_"
    PARTNER_FILENAME = "partner_"

    def __init__(self, data_reader: DataReader, model_trainer: ModelTrainer, model_dir: str):
        two_class = False
        if data_reader.partner_filename != "":
            two_class = True
        super().__init__(data_reader, model_trainer, two_class)

        self.model_dir = model_dir

    def train_models(self):
        self.train_user_model()

        if self.two_class:
            self.train_partner_model()

    def train_user_model(self):
        users_df, user_a_map, user_b_map = self.data_reader.read_user_data()

        self.model_trainer.reset_data()
        self.model_trainer.setup_data(users_df, len(user_a_map), len(user_b_map))

        self.save_model(self.USER_FILENAME)

    def train_partner_model(self):
        partners_df, partner_a_map, partner_b_map = self.data_reader.read_partner_data()

        self.model_trainer.reset_data()
        self.model_trainer.setup_data(partners_df, len(partner_a_map), len(partner_b_map))

        self.save_model(self.PARTNER_FILENAME)

    def save_model(self, filename):
        u, v, mse_list = self.model_trainer.train()

        filename_prefix = "%s%s" % (self.model_dir, filename)
        np.savetxt("%su.csv" % filename_prefix, u, delimiter=',')
        np.savetxt("%sv.csv" % filename_prefix, v, delimiter=',')