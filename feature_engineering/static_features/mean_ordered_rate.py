import pandas as pd

from feature_engineering.feature import StaticFeature


class MeanOrderedRate(StaticFeature):

    def __init__(self):
        super().__init__('mean_ordered_rate')
        self.feature_type = self.STEADY_FEATURE

    def _compute_feature(self):
        orders_joined_sorted = self.orders_joined.sort_values(by=['user_id', 'order_number'])
        orders_joined_sorted['num_prev_ordered'] = orders_joined_sorted.groupby(['user_id', 'product_id']).cumcount()
        # compute only for all previous orders
        orders_joined_sorted['ordered_rate'] = orders_joined_sorted['num_prev_ordered'] / (
                orders_joined_sorted['order_number'] - 1)
        mean_ordered_rate = orders_joined_sorted.groupby('order_id')['ordered_rate'].mean().reset_index()
        mean_ordered_rate = mean_ordered_rate.rename(columns={'ordered_rate': self.feature})
        self.orders_tip = pd.merge(self.orders_tip, mean_ordered_rate, on='order_id', how='left')
