from feature_engineering import DynamicFeature


class DynamicFeatureTest1(DynamicFeature):

    def __init__(self, data_store):
        super().__init__(data_store, 'dynamic_feature_test_1')

    def _compute_feature(self):
        self.orders_tip[self.feature] = self.orders_tip['order_number'] % 2

    def _analyze_feature(self):
        pass
