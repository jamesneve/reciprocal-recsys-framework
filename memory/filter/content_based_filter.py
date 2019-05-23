from .filter import Filter


class ContentBasedFilter(Filter):

    def __init__(self, preference_aggregator, user_preferences_df, user_attributes_df):
        super().__init__(preference_aggregator)
        self.user_preferences_df = user_preferences_df
        self.user_attributes_df = user_attributes_df
