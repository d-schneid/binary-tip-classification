from feature_engineering import StaticFeature


class HodHighTipProbability(StaticFeature):

    def __init__(self):
        super().__init__('hod_high_tip_probability')

    def _compute_feature(self):
        hour_of_day_high_tip_probability = [0, 1, 2, 3, 4, 19, 20, 21, 22, 23]

        self.orders_tip[self.feature] = self.orders_tip['order_hour_of_day'].apply(
            lambda x: 1 if x in hour_of_day_high_tip_probability else 0)

    def _analyze_feature(self):
        pass
