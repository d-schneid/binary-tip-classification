import numpy as np

from feature_engineering import StaticFeature


class PrevOrderTipped(StaticFeature):

    def __init__(self):
        super().__init__('prev_order_tipped')
        self.feature_type = self.BINARY_FEATURE

    def _compute_feature(self):
        self.orders_tip[self.feature] = self.orders_tip.groupby('user_id')['tip'].shift(1, fill_value=np.nan)