from abc import abstractmethod

import pandas as pd

from feature_engineering import DynamicFeature


class TipRate(DynamicFeature):

    def __init__(self, name, id_name):
        super().__init__(name)
        self._id = id_name

    def _compute_feature(self):
        orders_joined_copy = self.orders_joined.copy()
        orders_joined_all = self.orders_joined

        product_tip_prob = orders_joined_copy.dropna(subset=['tip']).groupby(self._id)['tip'].mean().reset_index()
        product_tip_prob.columns = [self._id, 'tip_probability']

        orders_products_tip_prob = pd.merge(orders_joined_all, product_tip_prob, on=self._id, how='left')
        product_tip_rate_by_order = orders_products_tip_prob.groupby('order_id')['tip_probability'].mean().reset_index()
        product_tip_rate_by_order.columns = ['order_id', self.feature]

        self.orders_tip = pd.merge(self.orders_tip, product_tip_rate_by_order, on='order_id', how='left')

    @abstractmethod
    def _analyze_feature(self):
        pass


class ProductTipRate(TipRate):

    def __init__(self):
        super().__init__('product_tip_rate', 'product_id')

    def _analyze_feature(self):
        pass


class DepartmentTipRate(TipRate):

    def __init__(self):
        super().__init__('dept_tip_rate', 'department_id')

    def _analyze_feature(self):
        pass


class AisleTipRate(TipRate):

    def __init__(self):
        super().__init__('aisle_tip_rate', 'aisle_id')

    def _analyze_feature(self):
        pass
