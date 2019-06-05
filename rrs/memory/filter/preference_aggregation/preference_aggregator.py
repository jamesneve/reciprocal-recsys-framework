import numpy as np


class PreferenceAggregator(object):

    def __init__(self, ab_score, ba_score):
        self.ab_score = ab_score
        self.ba_score = ba_score

    def aggregate_scores(self):
        pass
