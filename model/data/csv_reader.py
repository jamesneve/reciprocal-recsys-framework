import pandas as pd
from . import data_reader


class CsvReader(data_reader.DataReader):

    def read_data(self):
        users_df = pd.read_csv(self.filename)
        self.validate_dataframe(users_df)
        return users_df
