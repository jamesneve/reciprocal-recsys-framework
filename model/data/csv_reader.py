import pandas as pd
from . import data_reader


class CsvReader(data_reader.DataReader):

    def read_user_data(self):
        users_df = pd.read_csv(self.user_filename, dtype={'user_a': int, 'user_b': int, 'score': float})
        self.validate_dataframe(users_df)

        users_df, user_a_map, user_b_map = self.dataframe_to_user_map(users_df)
        return users_df, user_a_map, user_b_map

    def read_partner_data(self):
        partners_df = pd.read_csv(self.partner_filename, dtype={'user_a': int, 'user_b': int, 'score': float})
        self.validate_dataframe(partners_df)

        partners_df, partner_a_map, partner_b_map = self.dataframe_to_user_map(partners_df)
        return partners_df, partner_a_map, partner_b_map

