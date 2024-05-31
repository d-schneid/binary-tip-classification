import numpy as np
import pandas as pd

from feature_engineering.feature import StaticFeature


class ReorderedRatio(StaticFeature):

    def __init__(self):
        super().__init__('reordered_ratio')

    def _compute_feature(self):
        reordered_rate = (self.orders_joined.groupby('order_id')['reordered'].mean().reset_index()
                          .rename(columns={'reordered': self.feature}))
        self.orders_tip = pd.merge(self.orders_tip, reordered_rate, on='order_id', how='left')

        self.orders_tip.loc[self.orders_tip['order_number'] == 1, self.feature] = np.nan

    def _analyze_feature(self):
        pass
