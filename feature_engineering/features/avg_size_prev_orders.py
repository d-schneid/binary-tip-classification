# average #products in previous orders of the respective user compared to current order
import pandas as pd

from feature_engineering.feature import StaticFeature


class AvgSizePrevOrders(StaticFeature):

    def __init__(self, data_store):
        super().__init__(data_store, 'avg_size_prev_orders')

    def _compute_feature(self):
        order_size = (self.orders_joined.groupby('order_id')['order_number'].size().reset_index()
                      .rename(columns={'order_number': 'order_size'}))
        base = pd.merge(self.orders_joined, order_size, on='order_id', how='left')
        avg_size_prev_orders = (
            base.groupby('user_id')['order_size'].mean().reset_index()
            .rename(columns={'order_size': self.feature}))
        latest_order = base[base['order_number'] == base.groupby('user_id')['order_number'].transform('max')]
        latest_order = latest_order[['order_id', 'user_id']]
        # latest_order unique rows
        latest_order = latest_order.drop_duplicates()
        # compare average size of previous orders to current order
        result = pd.merge(latest_order, avg_size_prev_orders, on='user_id', how='left')
        result['avg_size_prev_orders'] = result['avg_size_prev_orders'].fillna(0)
        result = pd.merge(result, base[['order_id', 'order_size']], on='order_id', how='left')
        result['avg_size_prev_orders'] = result['avg_size_prev_orders'] / result['order_size']
        result.drop(['order_size'], axis=1, inplace=True)
        result.drop(['order_id'], axis=1, inplace=True)
        result = result.drop_duplicates()
        result = result.rename(columns={'avg_size_prev_orders': self.feature})

        self.orders_tip = pd.merge(self.orders_tip, result, on='user_id', how='left')

    def _analyze_feature(self):
        pass
