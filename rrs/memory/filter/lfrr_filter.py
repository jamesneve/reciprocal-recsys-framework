from .collaborative_filter import CollaborativeFilter
from .preference_aggregation import PreferenceAggregator
from ...data.data_reader import DataReader
import numpy as np
import pandas as pd


class LfrrFilter(CollaborativeFilter):

    def __init__(self, preference_aggregator: PreferenceAggregator, data_reader: DataReader):
        users_df, user_a_map, user_b_map = data_reader.read_user_data()
        self.users_df = users_df
        self.user_a_map = user_a_map
        self.user_b_map = user_b_map

        partners_df, partner_a_map, partner_b_map = data_reader.read_partner_data()
        self.partners_df = partners_df
        self.partner_a_map = partner_a_map
        self.partner_b_map = partner_b_map

        super().__init__(preference_aggregator, users_df)
        a_b_u, a_b_v = self.load_model(self.model_dir, self.A_B_PREFIX)

        self.a_b_u = a_b_u
        self.a_b_v = a_b_v

        b_a_u, b_a_v = self.load_model(self.model_dir, self.B_A_PREFIX)
        self.b_a_u = b_a_u
        self.b_a_v = b_a_v

    def predict(self, a_b_user_a: int, a_b_user_b: int, b_a_user_a: int = -1, b_a_user_b: int = -1):
        assert (a_b_user_a in self.a_b_u and a_b_user_b in self.a_b_v), "User IDs not in the training set"

        a_b_score = np.dot(self.a_b_u[a_b_user_a], self.a_b_v[a_b_user_b])
        assert (b_a_user_b in self.b_a_u and b_a_user_a in self.b_a_v), "User IDs not in the training set"

        b_a_score = np.dot(self.b_a_u[b_a_user_b], self.b_a_v[b_a_user_a])

        reciprocal_score = self.preference_aggregator.aggregate_scores(a_b_score, b_a_score)

        return reciprocal_score

    def load_model(self, model_dir, prefix):
        u = np.loadtxt("%s%su.csv" % (model_dir, prefix), delimiter=',')
        v = np.savetxt("%s%sv.csv" % (model_dir, prefix), delimiter=',')
        return u, v
