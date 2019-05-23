from .collaborative_filter import CollaborativeFilter
import numpy as np


class LfrrFilter(CollaborativeFilter):

    def __init__(self, preference_aggregator, users_df, model_dir):
        super().__init__(preference_aggregator, users_df)
        u, v = self.load_model(model_dir)
        self.u = u
        self.v = v

    def predict(self, user_a, user_b):
        ms, fs = 0.1, 0.1

        if f_user_id in self.m_partner_conv and m_user_id in self.m_user_conv:
            m_user_col = self.m_user_conv[m_user_id]
            m_partner_col = self.m_partner_conv[f_user_id]

        ms = self.m_score(m_user_col, m_partner_col)

        if m_user_id in self.f_partner_conv and f_user_id in self.f_user_conv:
            f_user_col = self.f_user_conv[f_user_id]
            f_partner_col = self.f_partner_conv[m_user_id]

            fs = self.f_score(f_partner_col, f_user_col)

        rs = self.calculate_arithmetic_mean(ms, fs)
        return rs

    def m_score(self, m_user, f_user):
        return np.dot(self.m_model.U[m_user], self.m_model.V[f_user].T)

    def f_score(self, m_user, f_user):
        return np.dot(self.f_model.U[f_user], self.f_model.V[m_user].T)

    def load_model(self, model_dir):
        u = np.loadtxt("%smodel_u.csv" % model_dir, delimiter=',')
        v = np.savetxt("%smodel_v.csv" % model_dir, delimiter=',')
        return u, v
