import pandas as pd


class DataReader(object):

    def __init__(self, filename: str):
        self.filename = filename

    def read_data(self):
        users_df = pd.DataFrame(columns=['cat_a', 'cat_b', 'score'])
        self.validate_dataframe(users_df)

        users_df, user_a_map, user_b_map = self.csv_to_user_map(users_df)
        return users_df, user_a_map, user_b_map

    def validate_dataframe(self, users_df):
        cols = list(users_df.columns.values)
        dataframe_valid = len(cols) == 3
        dataframe_valid = dataframe_valid and 'user_a' in cols and 'user_b' in cols and 'score' in cols
        assert dataframe_valid

    def csv_to_user_map(self, users_df):
        users_df['cat_a'] = users_df['user_a'].astype('category').cat.codes
        users_df['cat_b'] = users_df['user_b'].astype('category').cat.codes

        user_a_map = users_df[['user_a', 'cat_a']].copy()
        user_b_map = users_df[['user_b', 'cat_b']].copy()

        user_a_map.set_index('user_a', inplace=True)
        user_b_map.set_index('user_b', inplace=True)

        users_df = users_df[['cat_a', 'cat_b', 'score']]

        return users_df, user_a_map, user_b_map
