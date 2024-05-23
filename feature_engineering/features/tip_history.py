from feature_engineering import StaticFeature


class TipHistory(StaticFeature):

    def __init__(self, data_store):
        super().__init__(data_store, 'tip_history')

    def _compute_feature(self):
        self.orders_tip[self.feature] = (self.orders_tip.assign(tip_bool=self.orders_tip['tip'].astype(bool))
                                         .groupby('user_id')['tip_bool']
                                         .transform('cumsum').shift(1) / self.orders_tip['order_number'].shift(1))
        self.orders_tip.loc[self.orders_tip['order_number'] == 1, self.feature] = -1

    def _analyze_feature(self):
        pass
