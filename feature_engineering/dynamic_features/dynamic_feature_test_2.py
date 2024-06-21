import pandas as pd

from feature_engineering import DynamicFeature


class DynamicFeatureTest2(DynamicFeature):

    def __init__(self):
        super().__init__('dynamic_feature_test_2')
        self.feature_type = self.STEADY_FEATURE

    def _compute_feature(self):
        dynamic_feature_test_2 = self.orders_tip[['order_id', 'user_id', 'order_number']].copy()
        dynamic_feature_test_2[self.feature] = self.orders_tip.groupby('user_id')['order_number'].transform('rank') / \
                                               self.orders_tip.groupby('user_id')['order_number'].transform('size')
        self.orders_tip = pd.merge(self.orders_tip, dynamic_feature_test_2[[self.feature, 'order_id']], on='order_id',
                                   how='left')