from feature_engineering import StaticFeature


class DowHighTipProbability(StaticFeature):

    def __init__(self):
        super().__init__('dow_high_tip_probability')
        self.feature_type = self.BINARY_FEATURE

    def _compute_feature(self):
        dow_high_tip_probability = [0, 1]

        self.orders_tip[self.feature] = self.orders_tip['order_dow'].apply(
            lambda x: 1 if x in dow_high_tip_probability else 0)
