import numpy as np
from .preference_aggregation import PreferenceAggregator


class Filter(object):

    def __init__(self, preference_aggregator: PreferenceAggregator):
        self.preference_aggregator = preference_aggregator

    def predict(self, user_a: int, user_b: int):
        pass
