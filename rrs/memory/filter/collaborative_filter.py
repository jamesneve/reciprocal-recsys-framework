from .filter import Filter
from .preference_aggregation import PreferenceAggregator
from ...data.data_reader import DataReader


class CollaborativeFilter(Filter):

    def __init__(self, preference_aggregator: PreferenceAggregator, reader: DataReader):
        super().__init__(preference_aggregator)
        users_df, _, _ = reader.read_user_data()
        self.users_df = users_df
