from feature_engineering import StaticFeature


class ContainsAlcohol(StaticFeature):

    def __init__(self):
        super().__init__('contains_alcohol')

    def _compute_feature(self):
        order_ids_with_alcohol = self.orders_joined[self.orders_joined['department_id'] == 5]['order_id'].unique()
        self.orders_tip[self.feature] = self.orders_tip['order_id'].isin(order_ids_with_alcohol).astype(int)

    def _analyze_feature(self):
        pass
