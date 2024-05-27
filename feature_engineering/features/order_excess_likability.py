# sum(reordered rate - mean(reordered rate)) across all products in the order of the respective user --> how much does user like order?

import pandas as pd

from feature_engineering.feature import StaticFeature


class OrderExcessLikability(StaticFeature):
    def __init__(self, data_store):
        super().__init__(data_store, 'order_excess_likability')

    def _compute_feature(self):
        reordered_rate = (self.orders_joined.groupby('order_id')['reordered'].mean().reset_index()
                          .rename(columns={'reordered': self.feature}))
        self.orders_tip = pd.merge(self.orders_tip, reordered_rate, on='order_id', how='left')

    def _analyze_feature(self):
        pass
