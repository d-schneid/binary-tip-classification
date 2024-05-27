import pandas as pd

from feature_engineering.feature import StaticFeature


class SumDaysSincePriorOrder(StaticFeature):

    def __init__(self):
        super().__init__('sum_days_since_prior_order')

    def _compute_feature(self):
        sum_days_since_prior_order = (
            self.orders_joined.groupby('user_id')['days_since_prior_order'].sum().reset_index()
            .rename(columns={'days_since_prior_order': self.feature}))
        self.orders_tip = pd.merge(self.orders_tip, sum_days_since_prior_order, on='user_id', how='left')

    def _analyze_feature(self):
        pass
