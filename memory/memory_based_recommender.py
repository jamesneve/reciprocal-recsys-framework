import numpy as np
from .data.data_reader import DataReader


class MemoryBasedRecommender(object):

    def __init__(self, data_reader: DataReader):
        self.data_reader = data_reader
