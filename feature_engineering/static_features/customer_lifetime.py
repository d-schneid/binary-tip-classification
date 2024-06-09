from feature_engineering import StaticFeature


class CustomerLifetime(StaticFeature):

    def __init__(self):
        super().__init__('customer_lifetime')

    def _compute_feature(self):
        self.orders_tip[self.feature] = self.orders_tip[self.orders_tip['order_number'] != 1].groupby('user_id')[
            'days_since_prior_order'].cumsum()
        self.orders_tip[self.feature] = self.orders_tip[self.feature].fillna(0).astype(int)

    def _analyze_feature(self):
        pass
