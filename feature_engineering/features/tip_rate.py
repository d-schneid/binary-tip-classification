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

        # drop all orders that have no tip (tip==None)
        orders_joined_copy = orders_joined_copy.dropna(subset=['tip'])

        # change all tip==0 values to -1 and put them in a new column
        orders_joined_copy['tip_transformed'] = orders_joined_copy['tip'].apply(lambda x: -1 if x == 0 else 1)

        # calculate the tip_rate of every single product on the entire subset
        product_tip_rate = orders_joined_copy.groupby(self._id)['tip_transformed'].mean().reset_index()
        product_tip_rate.columns = [self._id, 'tip_rate']

        # join the specific product tip rate to the orders_joined table
        orders_products_tip_rate = pd.merge(self.orders_joined, product_tip_rate, on=self._id, how='left')

        # in case products has not been seen in the calculation of the product_tip_rate assign neutral value 0
        orders_products_tip_rate['tip_rate'] = orders_products_tip_rate['tip_rate'].fillna(0)

        # calculate the product_tip_rate of the order based on the mean of the product_tip_rate for every product
        product_tip_rate_by_order = orders_products_tip_rate.groupby('order_id')['tip_rate'].mean().reset_index()

        # merge feature into overall dataset
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
