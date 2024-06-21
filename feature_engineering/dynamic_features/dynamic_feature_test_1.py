from feature_engineering import DynamicFeature


class DynamicFeatureTest1(DynamicFeature):

    def __init__(self):
        super().__init__('dynamic_feature_test_1')
        self.feature_type = self.STEADY_FEATURE

    def _compute_feature(self):
        self.orders_tip[self.feature] = self.orders_tip.groupby('user_id')['order_number'].transform('rank') / \
                                        self.orders_tip.groupby('user_id')['order_number'].transform('size')
