import pandas as pd

from feature_engineering import DynamicFeature


class DynamicFeatureTest2(DynamicFeature):

    def __init__(self, data_store):
        super().__init__(data_store, 'dynamic_feature_test_2')

    def _compute_feature(self):
        # Compute the modulo of the order number but keep the order_id
        dynamic_feature_test_2 = self.orders_tip[['order_id', 'order_number']].copy()
        dynamic_feature_test_2[self.feature] = dynamic_feature_test_2['order_number'] % 3
        self.orders_tip = pd.merge(self.orders_tip, dynamic_feature_test_2[[self.feature, 'order_id']], on='order_id',
                                   how='left')

    def _analyze_feature(self):
        pass
