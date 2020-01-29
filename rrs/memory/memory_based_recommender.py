from ..data.data_reader import DataReader
from .filter import Filter
from .hybrid_aggregation.hybrid_aggregator import HybridAggregator

from typing import List


class MemoryBasedRecommender(object):

    def __init__(self, data_reader: DataReader, filters: List[Filter], hybrid_aggregator: HybridAggregator):
        self.data_reader = data_reader
        self.filters = filters
        self.hybrid_aggregator = hybrid_aggregator
        if data_reader.partner_filename != "":
            self.two_class = True

    def generate_reciprocal_score(self, user_a, user_b):
        a_b_users_df, user_a_map, user_b_map = self.data_reader.read_user_data()

        assert (user_a in user_a_map), "User A is not in the dataset"
        assert (user_b in user_b_map), "User B is not in the dataset"

        a_b_user_a, b_a_user_a = user_a_map[user_a], -1
        a_b_user_b, b_a_user_b = user_b_map[user_b], -1

        b_a_users_df, b_a_user_a_map, b_a_user_b_map = self.data_reader.read_partner_data()

        assert (user_a in b_a_user_a_map), "User A is not in the reciprocal dataset"
        assert (user_b in b_a_user_b_map), "User B is not in the reciprocal dataset"

        b_a_user_a = b_a_user_a_map[user_a]
        b_a_user_b = b_a_user_b_map[user_b]

        filter_outputs = []
        for filter in self.filters:
            score = filter.predict(a_b_user_a, a_b_user_b, b_a_user_a, b_a_user_b)
            filter_outputs.append(score)

        hybrid_score = self.hybrid_aggregator.aggregate_scores(filter_outputs)

        return hybrid_score
