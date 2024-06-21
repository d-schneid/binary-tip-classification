from feature_engineering import StaticFeature


class OrderNumberSquared(StaticFeature):

    def __init__(self):
        super().__init__('order_number_squared')
        self.feature_type = self.STEADY_FEATURE

    def _compute_feature(self):
        self.orders_tip[self.feature] = self.orders_tip['order_number'] ** 2
