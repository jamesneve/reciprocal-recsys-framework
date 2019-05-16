import pandas as pd
import numpy as np

def LfrrTrainer(ModelTrainer):

    def __init__(self, users_df, user_map, k, alpha, iterations, lmda):
        self.alpha = alpha
        self.R = users_df
        self.U = np.zeros((len(user_map), k,))
        self.V = np.zeros((len(user_map), k,))
        self.iterations = iterations
        self.lmda = lmda

    def uniform_initialize_uv(self):
        self.U = np.random.normal(0, 1.0, np.shape(self.U))
        self.V = np.random.normal(0, 1.0, np.shape(self.V))

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
            x = int(row['user_id'])
            y = int(row['partner_id'])
            v = row['score']
            predicted_v = np.dot(self.U[x], self.V[y].T)
            error = v - predicted_v
            mse += np.power(error, 2)
        return mse/float(len(self.R))

    def gradient_descent(self):
        df = self.R.sample(frac=1)
        for index, row in df.iterrows():
            i = int(row['user_id'])
            j = int(row['partner_id'])
            v = row['score']
            prediction = np.dot(self.U[i, :], self.V[j, :].T)
            e = v - prediction

            self.U[i, :] += self.alpha * (e * self.V[j, :] - self.lmda * self.U[i, :])
            self.V[j, :] += self.alpha * (e * self.U[i, :] - self.lmda * self.V[j, :])