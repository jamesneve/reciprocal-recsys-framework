from .filter import Filter


class CollaborativeFilter(Filter):

    def __init__(self, preference_aggregator, users_df):
        super().__init__(preference_aggregator)
        self.users_df = users_df
