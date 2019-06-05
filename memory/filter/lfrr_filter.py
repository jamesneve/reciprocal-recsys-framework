from .collaborative_filter import CollaborativeFilter
from .preference_aggregation import PreferenceAggregator
import numpy as np
import pandas as pd


class LfrrFilter(CollaborativeFilter):

    DEFAULT_SCORE = 0.1

    def __init__(self, preference_aggregator: PreferenceAggregator, users_df: pd.DataFrame, model_dir: str,
                 filename_prefix: str):
        super().__init__(preference_aggregator, users_df)
        u, v = self.load_model(model_dir, filename_prefix)

        self.u = u
        self.v = v

    def predict(self, user_a, user_b):
        if user_a not in self.u or user_b not in self.v:
            return self.DEFAULT_SCORE

        score = np.dot(self.u[user_a], self.v[user_b])

        return score

    def load_model(self, model_dir, prefix):
        u = np.loadtxt("%s%su.csv" % (model_dir, prefix), delimiter=',')
        v = np.savetxt("%s%sv.csv" % (model_dir, prefix), delimiter=',')
        return u, v
