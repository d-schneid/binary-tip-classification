import pandas as pd
from matplotlib import pyplot as plt

from analysis import Analysis


class ProductOrderPosition(Analysis):

    def __init__(self, data_manager):
        super().__init__(data_manager)

    def _analyze(self):

        product_tip_rate_card_order = (self.orders_joined.groupby('tip_rate_card_order')
                                       .apply(self._calculate_card_order_tip_rate, include_groups=False)
                                       .reset_index())



    def _show_results(self):
        pass

    def _calculate_card_order_tip_rate(self, product_orders):
        tip_rate_card_order = product_orders.groupby('add_to_cart_order')['tip'].mean()

        return tip_rate_card_order.std()



