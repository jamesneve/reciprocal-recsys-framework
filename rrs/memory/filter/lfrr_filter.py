from .collaborative_filter import CollaborativeFilter
from .preference_aggregation import PreferenceAggregator
import numpy as np
import pandas as pd


class LfrrFilter(CollaborativeFilter):

    DEFAULT_SCORE = 0.1

    def __init__(self, preference_aggregator: PreferenceAggregator, users_df: pd.DataFrame, model_dir: str,
                 filename_prefix: str, partner_model_dir: str = ""):
        super().__init__(preference_aggregator, users_df)
        self.two_class = (partner_model_dir != "")
        a_b_u, a_b_v = self.load_model(model_dir, filename_prefix)

        self.a_b_u = a_b_u
        self.a_b_v = a_b_v

        if self.two_class:
            b_a_u, b_a_v = self.load_model(partner_model_dir, filename_prefix)
            self.b_a_u = b_a_u
            self.b_a_v = b_a_v

    def predict(self, a_b_user_a: int, a_b_user_b: int, b_a_user_a: int = -1, b_a_user_b: int = -1):
        assert (a_b_user_a in self.a_b_u and a_b_user_b in self.a_b_v), "User IDs not in the training set"

        a_b_score = np.dot(self.a_b_u[a_b_user_a], self.a_b_v[a_b_user_b])
        if self.two_class:
            assert (b_a_user_b in self.b_a_u and b_a_user_a in self.b_a_v), "User IDs not in the training set"

            b_a_score = np.dot(self.b_a_u[b_a_user_b], self.b_a_v[b_a_user_a])
        else:
            assert (a_b_user_b in self.a_b_u and a_b_user_a in self.a_b_v), "User IDs not in the training set"
            b_a_score = np.dot(self.a_b_u[a_b_user_b], self.a_b_v[a_b_user_a])

        reciprocal_score = self.preference_aggregator.aggregate_scores(a_b_score, b_a_score)

        return reciprocal_score

    def load_model(self, model_dir, prefix):
        u = np.loadtxt("%s%su.csv" % (model_dir, prefix), delimiter=',')
        v = np.savetxt("%s%sv.csv" % (model_dir, prefix), delimiter=',')
        return u, v
