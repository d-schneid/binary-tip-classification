from feature_engineering import StaticFeature


class OrderNumberSquared(StaticFeature):

    def __init__(self):
        super().__init__('order_number_squared')

    def _compute_feature(self):
        self.orders_tip[self.feature] = self.orders_tip['order_number'] ** 2

    def _analyze_feature(self):
        pass
