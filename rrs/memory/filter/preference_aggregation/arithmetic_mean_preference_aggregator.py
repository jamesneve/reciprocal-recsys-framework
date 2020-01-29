import numpy as np
from .preference_aggregator import PreferenceAggregator


class ArithmeticMeanPreferenceAggregator(PreferenceAggregator):

    def aggregate_scores(self, a_b_score: float, b_a_score: float):
        res = self.arithmetic_mean(a_b_score, b_a_score)
        return res

    def arithmetic_mean(self, x, y):
        am = (x + y) / 2.0
        return am
