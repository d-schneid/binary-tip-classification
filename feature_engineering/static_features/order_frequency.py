import numpy as np

from feature_engineering import StaticFeature


class OrderFrequency(StaticFeature):

    def __init__(self):
        super().__init__('order_frequency')
        self.feature_type = self.STEADY_FEATURE

    def _compute_feature(self):
        # ensure NaN for accurate computation
        self.orders_tip.loc[self.orders_tip['order_number'] == 1, 'days_since_prior_order'] = np.nan
        self.orders_tip[self.feature] = (self.orders_tip.groupby('user_id')['days_since_prior_order'].expanding()
                                         .mean().reset_index(level=0, drop=True))
        self.orders_tip.loc[self.orders_tip['order_number'] == 1, self.feature] = np.nan

