import pandas as pd

from feature_engineering import StaticFeature


class OrderSize(StaticFeature):

    def __init__(self, data_store):
        super().__init__(data_store, 'order_size')

    def _compute_feature(self):
        order_size = (self.orders_joined.groupby('order_id')['order_number'].size().reset_index()
                      .rename(columns={'order_number': self.feature}))
        self.orders_tip = pd.merge(self.orders_tip, order_size, on='order_id', how='left')

    def _analyze_feature(self):
        pass
