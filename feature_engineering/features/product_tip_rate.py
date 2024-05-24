from feature_engineering import DynamicFeature
import pandas as pd


class ProductTipRate(DynamicFeature):

    def __init__(self, data_store):
        super().__init__(data_store, 'product_tip_rate')

    def _compute_feature(self):
        orders_joined_copy = self.orders_joined.copy()
        orders_joined_all = self.orders_joined

        product_tip_prob = orders_joined_copy.dropna(subset=['tip']).groupby('product_id')['tip'].mean().reset_index()
        product_tip_prob.columns = ['product_id', 'tip_probability']

        orders_products_tip_prob = pd.merge(orders_joined_all, product_tip_prob, on='product_id', how='left')
        product_tip_rate_by_order = orders_products_tip_prob.groupby('order_id')['tip_probability'].mean().reset_index()
        product_tip_rate_by_order.columns = ['order_id', self.feature]

        self.orders_tip = pd.merge(self.orders_tip, product_tip_rate_by_order, on='order_id', how='left')

    def _analyze_feature(self):
        pass
