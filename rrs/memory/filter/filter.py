import numpy as np
from .preference_aggregation import PreferenceAggregator
import os


class Filter(object):

    A_B_PREFIX = os.environ['A_B_PREFIX']
    B_A_PREFIX = os.environ['B_A_PREFIX']

    def __init__(self, preference_aggregator: PreferenceAggregator, model_dir: str = ""):
        self.preference_aggregator = preference_aggregator
        self.model_dir = model_dir

    def predict(self, a_b_user_a: int, a_b_user_b: int, b_a_user_a: int = -1, b_a_user_b: int = -1):
        pass
