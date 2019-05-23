import abc

class ModelTrainer(object):

    @abc.abstractmethod
    def setup_data(self, users_df, len_user_a, len_user_b):
        pass

    @abc.abstractmethod
    def reset_data(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass
