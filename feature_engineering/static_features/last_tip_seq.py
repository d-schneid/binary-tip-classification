import math

import pandas as pd
import numpy as np

from feature_engineering import StaticFeature


class LastTipSequence(StaticFeature):

    def __init__(self):
        super().__init__('last_tip_sequence')

    def _compute_feature(self):
        calculated_tip_sequence = (self.orders_tip.groupby('user_id')
                                   .apply(self._calculate_last_tip_sequence, include_groups=False)
                                   .reset_index(drop=False))

        self.orders_tip = pd.merge(self.orders_tip, calculated_tip_sequence[['order_id', self.feature]]
                                   , on='order_id', how='left')

    def _calculate_last_tip_sequence(self, user_orders):
        user_orders = user_orders.sort_values(by=['order_number'], ascending=True)
        last_tip_sequence = 0

        for index, order in user_orders.iterrows():
            if order['order_number'] == 0:
                user_orders.at[index, self.feature] = np.nan
            else:
                user_orders.at[index, self.feature] = last_tip_sequence

            if order['tip'] == 1.0:
                last_tip_sequence += 1
            else:
                last_tip_sequence = 0

        return user_orders

    def _analyze_feature(self):
        pass
