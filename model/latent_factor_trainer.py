from .data.data_reader import DataReader
from .trainer.model_trainer import ModelTrainer
from .model_based_trainer import ModelBasedTrainer
import numpy as np


class LatentFactorTrainer(ModelBasedTrainer):

    def train_model(self, model_dir):
        users_df, user_a_map, user_b_map = self.data_reader.read_data()

        self.model_trainer.setup_data(users_df, len(user_a_map), len(user_b_map))
        u, v, mse_list = self.model_trainer.train()

        np.savetxt("%smodel_u.csv" % model_dir, u, delimiter=',')
        np.savetxt("%smodel_v.csv" % model_dir, v, delimiter=',')
        user_a_map.to_csv("%suser_a_map.csv" % model_dir)
        user_b_map.to_csv("%suser_b_map.csv" % model_dir)
