from .filter import Filter
from .preference_aggregation import PreferenceAggregator
import pandas as pd


class CollaborativeFilter(Filter):

    def __init__(self, preference_aggregator: PreferenceAggregator, users_df: pd.DataFrame):
        super().__init__(preference_aggregator)
        self.users_df = users_df
