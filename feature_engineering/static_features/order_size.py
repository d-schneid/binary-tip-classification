import pandas as pd

from feature_engineering import StaticFeature


class OrderSize(StaticFeature):

    def __init__(self):
        super().__init__('order_size')
        self.feature_type = self.DISCRETE_FEATURE

    def _compute_feature(self):
        order_size = (self.orders_joined.groupby('order_id')['order_number'].size().reset_index()
                      .rename(columns={'order_number': self.feature}))
        self.orders_tip = pd.merge(self.orders_tip, order_size, on='order_id', how='left')
