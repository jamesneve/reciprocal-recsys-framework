import pandas as pd


class DataReader(object):

    def __init__(self, filename: str):
        self.filename = filename

    def read_data(self):
        users_df = pd.DataFrame(columns=['user_a', 'user_b', 'score'])
        self.validate_dataframe(users_df)
        return users_df

    def validate_dataframe(self, users_df):
        cols = list(users_df.columns.values)
        dataframe_valid = len(cols) == 3 and 'user_a' in cols and 'user_b' in cols and 'score' in cols
        assert dataframe_valid
