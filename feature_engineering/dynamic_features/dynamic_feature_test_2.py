import pandas as pd

from feature_engineering import DynamicFeature


class DynamicFeatureTest2(DynamicFeature):

    def __init__(self):
        super().__init__('dynamic_feature_test_2')

    def _compute_feature(self):
        dynamic_feature_test_2 = self.orders_tip[['order_id', 'user_id', 'order_number']].copy()
        dynamic_feature_test_2[self.feature] = dynamic_feature_test_2['order_number'] / dynamic_feature_test_2.groupby(
            'user_id').transform('size')
        self.orders_tip = pd.merge(self.orders_tip, dynamic_feature_test_2[[self.feature, 'order_id']], on='order_id',
                                   how='left')

    def _analyze_feature(self):
        pass
