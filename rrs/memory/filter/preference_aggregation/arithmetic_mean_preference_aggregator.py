import numpy as np
from .preference_aggregator import PreferenceAggregator


class ArithmeticMeanPreferenceAggregator(PreferenceAggregator):

    def aggregate_scores(self):
        res = self.arithmetic_mean(self.ab_score, self.ba_score)
        return res

    def arithmetic_mean(self, x, y):
        am = (float(x) + float(y)) / 2.0
        return am
