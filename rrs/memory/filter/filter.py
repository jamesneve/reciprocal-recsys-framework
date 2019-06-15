import numpy as np
from .preference_aggregation import PreferenceAggregator


class Filter(object):

    def __init__(self, preference_aggregator: PreferenceAggregator):
        self.preference_aggregator = preference_aggregator

    def predict(self, a_b_user_a: int, a_b_user_b: int, b_a_user_a: int = -1, b_a_user_b: int = -1):
        pass
