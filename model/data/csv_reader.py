import pandas as pd
from . import data_reader


class CsvReader(data_reader.DataReader):

    def read_data(self):
        users_df = pd.read_csv(self.filename, dtype={'user_a': int, 'user_b': int, 'score': float})
        self.validate_dataframe(users_df)

        users_df, user_a_map, user_b_map = self.csv_to_user_map(users_df)
        return users_df, user_a_map, user_b_map
