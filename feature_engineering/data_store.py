import pandas as pd


class DataStore:
    def __init__(self, op_prior, op_train, tip_train, tip_test, orders, products, aisles, departments):
        # Initialize instance variables
        self.products = products
        self.aisles = aisles
        self.departments = departments
        self.tip_test = tip_test

        # Concatenate order products and tips
        self.op = pd.concat([op_prior, op_train])
        self.tip = pd.concat([tip_train, tip_test[['order_id', 'tip']]])
        self.orders_tip = pd.merge(orders, self.tip)

        # Prepare joined data
        self.orders_joined = self.prepare_data_for_processing()

        # Initialize subsets used for dynamic feature computation
        self.orders_tip_subset = None
        self.orders_joined_subset = None

    def prepare_data_for_processing(self):
        # Merge all necessary dataframes to create a final dataset for analysis
        orders_joined = pd.merge(self.orders_tip, self.op)
        orders_joined = pd.merge(orders_joined, self.products)
        orders_joined = pd.merge(orders_joined, self.aisles)
        return pd.merge(orders_joined, self.departments)

    def merge_orders_tip(self, orders_tip, feature):
        self.orders_tip = pd.merge(self.orders_tip, orders_tip[[feature, 'order_id']], on='order_id', how='left')

    def merge_orders_tip_subset(self, orders_tip, feature):
        self.orders_tip_subset = pd.merge(self.orders_tip_subset, orders_tip[[feature, 'order_id']], on='order_id',
                                          how='left')

    def get_orders_tip(self, columns=None):
        return self.orders_tip if columns is None else self.orders_tip[columns]

    def get_orders_joined(self, columns=None):
        return self.orders_joined if columns is None else self.orders_joined[columns]

    def get_orders_tip_subset(self, columns=None):
        return self.orders_tip_subset if columns is None else self.orders_tip_subset[columns]

    def get_orders_joined_subset(self, columns=None):
        return self.orders_joined_subset if columns is None else self.orders_joined_subset[columns]

    def set_data_subset(self, order_ids):
        self.orders_tip_subset = self.orders_tip[self.orders_tip['order_id'].isin(order_ids)]
        self.orders_joined_subset = self.orders_joined[self.orders_joined['order_id'].isin(order_ids)]
