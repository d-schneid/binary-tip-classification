import pandas as pd
import numpy as np
from feature_engineering import StaticFeature


class SimOrdersTipRatio(StaticFeature):

    def __init__(self):
        super().__init__('sim_orders_tip_ratio')

    def _compute_feature(self):
        order_with_grouped_products = (self.orders_joined.groupby(['user_id', 'order_number', 'order_id']).agg(
            products=('product_id', lambda x: set(x)), tip=('tip', 'first'))).reset_index()

        calculated_sim_tip_ratio = (order_with_grouped_products.groupby('user_id')
                                    .apply(self._calculate_similarity_tip_ratio, include_groups=False)
                                    .reset_index(drop=False))

        self.orders_tip = pd.merge(self.orders_tip, calculated_sim_tip_ratio[['order_id', self.feature]]
                                   , on='order_id', how='left')

    def _calculate_similarity_tip_ratio(self, user_orders):
        cumulative_products_tipped_orders = []
        cumulative_products_none_tipped_orders = []

        for index, order in user_orders.iterrows():
            comparison_results_tipped_orders = []
            comparison_results_none_tipped_orders = []

            if order['order_number'] == 1:
                user_orders.at[index, self.feature] = np.nan
            else:
                for order_products in cumulative_products_tipped_orders:
                    order_similarity = self._compare_orders_products_jaccard_similarity(order['products'], order_products)
                    comparison_results_tipped_orders.append(order_similarity)

                for order_products in cumulative_products_none_tipped_orders:
                    order_similarity = self._compare_orders_products_jaccard_similarity(order['products'], order_products)
                    comparison_results_none_tipped_orders.append(order_similarity)

                user_orders.at[index, self.feature] = (
                    (sum(comparison_results_tipped_orders) - sum(comparison_results_none_tipped_orders)) / order['order_number'] - 1
                )

            if order['tip'] == 1.0:
                cumulative_products_tipped_orders.append(order['products'])
            else:
                cumulative_products_none_tipped_orders.append(order['products'])

        return user_orders

    def _compare_orders_products_jaccard_similarity(self, products1, products2):
        intersection_size = len(products1.intersection(products2))
        union_size = len(products1.union(products2))
        return intersection_size / union_size if union_size != 0 else 0

    def _analyze_feature(self):
        pass
