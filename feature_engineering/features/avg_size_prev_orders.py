# average #products in previous orders of the respective user compared to current order
import numpy as np
import pandas as pd

from feature_engineering.feature import StaticFeature


class AvgSizePrevOrders(StaticFeature):

    def __init__(self):
        super().__init__('avg_size_prev_orders')

    def _compute_feature(self):
        orders = self.orders_joined.copy()
        orders = orders[['user_id', 'order_id', 'order_number']]
        orders = orders.sort_values(['user_id', 'order_number'])
        # Calculate the order size
        orders['order_size'] = orders.groupby('order_id')['order_number'].transform('size')
        orders = orders.drop_duplicates(subset=['user_id', 'order_number'], keep='last')
        # Calculate the average order size of previous orders
        orders['avg_prev'] = orders.groupby('user_id')['order_size'].cumsum() / orders['order_number']
        orders['avg_prev'] = orders.groupby('user_id')['avg_prev'].shift(1).fillna(0).astype(np.float32)
        orders[self.feature] = orders['avg_prev'] / orders['order_size']

        self.orders_tip = pd.merge(self.orders_tip, orders[['order_id', self.feature]], on='order_id', how='left')

    def _analyze_feature(self):
        pass
