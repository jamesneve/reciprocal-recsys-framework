import numpy as np
from .data.data_reader import DataReader
from .filter import Filter

from typing import List


class MemoryBasedRecommender(object):

    def __init__(self, data_reader: DataReader, filters: List[Filter]):
        self.data_reader = data_reader
        self.filters = filters
        self.two_class = False
        if data_reader.partner_filename != "":
            self.two_class = True

    def generate_reciprocal_score(self):
        users_df, user_a_map, user_b_map = self.data_reader.read_user_data()
