import pandas as pd
import numpy as np
from .model_trainer import ModelTrainer


class LfrrTrainer(ModelTrainer):

    def __init__(self, k, alpha, iterations, lmda):
        super().__init__()

        self.R = pd.DataFrame(columns=['cat_a', 'cat_b', 'score'])
        self.k = k
        self.alpha = alpha
        self.iterations = iterations
        self.lmda = lmda
        self.U = []
        self.V = []

    def setup_data(self, users_df, len_user_a, len_user_b):
        self.R = users_df
        self.U = np.zeros((len_user_a, self.k,))
        self.V = np.zeros((len_user_b, self.k,))

        self.U = np.random.normal(0, 1.0, np.shape(self.U))
        self.V = np.random.normal(0, 1.0, np.shape(self.V))

    def reset_data(self):
        self.R = pd.DataFrame(columns=['cat_a', 'cat_b', 'score'])
        self.U = []
        self.V = []

    def train(self):
        mse_list = [self.mse()]
        for i in range(0, self.iterations):
            self.gradient_descent()
            mse = self.mse()
            mse_list.append(mse)

        return self.U, self.V, mse_list

    def mse(self):
        mse = 0.0
        for index, row in self.R.iterrows():
            x = int(row['cat_a'])
            y = int(row['cat_b'])
            v = row['score']
            predicted_v = np.dot(self.U[x], self.V[y].T)
            error = v - predicted_v
            mse += np.power(error, 2)
        return mse/float(len(self.R))

    def gradient_descent(self):
        df = self.R.sample(frac=1)
        for index, row in df.iterrows():
            i = int(row['cat_a'])
            j = int(row['cat_b'])
            v = row['score']
            prediction = np.dot(self.U[i, :], self.V[j, :].T)
            e = v - prediction

            self.U[i, :] += self.alpha * (e * self.V[j, :] - self.lmda * self.U[i, :])
            self.V[j, :] += self.alpha * (e * self.U[i, :] - self.lmda * self.V[j, :])