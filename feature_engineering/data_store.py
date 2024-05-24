import pandas as pd


class DataStore:
    def __init__(self, op_prior, op_train, tip_train, tip_test, orders, products, aisles, departments):
        # Initialize instance variables
        self._products = products
        self._aisles = aisles
        self._departments = departments

        # Concatenate order products and tips
        self._op = pd.concat([op_prior, op_train])
        self._tip = pd.concat([tip_train, tip_test[['order_id', 'tip']]])
        self._orders_tip = pd.merge(orders, self._tip)

        # Prepare joined data
        self._orders_joined = self._prepare_data_for_processing()

        # Initialize subsets used for dynamic feature computation
        self._orders_tip_subset = self._orders_tip
        self._orders_joined_subset = self._orders_joined

    def _prepare_data_for_processing(self):
        # Merge all necessary dataframes to create a final dataset for analysis
        orders_joined = pd.merge(self._orders_tip, self._op)
        orders_joined = pd.merge(orders_joined, self._products)
        orders_joined = pd.merge(orders_joined, self._aisles)
        return pd.merge(orders_joined, self._departments)

    def merge_orders_tip(self, orders_tip, feature):
        self._orders_tip = pd.merge(self._orders_tip, orders_tip[[feature, 'order_id']], on='order_id', how='left')
        self.merge_orders_tip_subset(orders_tip, feature)

    def merge_orders_tip_subset(self, orders_tip, feature):
        self._orders_tip_subset = pd.merge(self._orders_tip_subset, orders_tip[[feature, 'order_id']], on='order_id',
                                           how='left')

    def get_orders_tip(self, columns=None):
        return self._orders_tip if columns is None else self._orders_tip[columns]

    def get_orders_joined(self, columns=None):
        return self._orders_joined if columns is None else self._orders_joined[columns]

    def get_orders_tip_subset(self, columns=None):
        return self._orders_tip_subset if columns is None else self._orders_tip_subset[columns]

    def get_orders_joined_subset(self, columns=None):
        return self._orders_joined_subset if columns is None else self._orders_joined_subset[columns]

    def set_data_subset(self, order_ids):
        self._orders_tip_subset = self._orders_tip[self._orders_tip['order_id'].isin(order_ids)].copy()
        self._orders_joined_subset = self._orders_joined[self._orders_joined['order_id'].isin(order_ids)].copy()
