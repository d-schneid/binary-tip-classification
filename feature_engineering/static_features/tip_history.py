import numpy as np

from feature_engineering import StaticFeature


class TipHistory(StaticFeature):

    def __init__(self):
        super().__init__('tip_history')

    def _compute_feature(self):
        self.orders_tip[self.feature] = (self.orders_tip.assign(tip_bool=self.orders_tip['tip'].astype(bool))
                                         .groupby('user_id')['tip_bool']
                                         .transform('cumsum').shift(1) / self.orders_tip['order_number'].shift(1))
        self.orders_tip.loc[self.orders_tip['order_number'] == 1, self.feature] = np.nan

    def _analyze_feature(self):
        pass
