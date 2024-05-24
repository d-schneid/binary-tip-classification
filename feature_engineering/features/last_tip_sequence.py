from feature_engineering import StaticFeature


class LastTipSequence(StaticFeature):

    def __init__(self, data_store):
        super().__init__(data_store, 'last_tip_sequence')

    def _compute_feature(self):
        self.orders_tip[self.feature] = self.orders_tip.apply(self._count_last_tips, axis=1)

    def _count_last_tips(self, order):
        consecutive_tips = 0
        for i in range(order['order_number'] - 1, 0, -1):

            if self.orders_tip[(self.orders_tip.user_id == order.user_id)
                               & (order.order_number == i), 'tip'].values[0] == 1:
                consecutive_tips += 1
            else:
                return consecutive_tips
        return consecutive_tips

    def _analyze_feature(self):
        pass
