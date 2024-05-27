import pandas as pd

from feature_engineering import StaticFeature, DynamicFeature


class DataManager:
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
        self._orders_tip_subset = self._orders_tip.copy()
        self._orders_joined_subset = self._orders_joined.copy()

        # Initialize feature sets
        self.dynamic_features = set()
        self.static_features = set()

    def _prepare_data_for_processing(self):
        # Merge all necessary dataframes to create a final dataset for analysis
        orders_joined = pd.merge(self._orders_tip, self._op)
        orders_joined = pd.merge(orders_joined, self._products)
        orders_joined = pd.merge(orders_joined, self._aisles)
        return pd.merge(orders_joined, self._departments)

    def get_orders_tip(self, complete=False):
        if complete:
            return self._orders_tip
        else:
            return self._orders_tip_subset

    def get_orders_joined(self, complete=False):
        if complete:
            return self._orders_joined
        else:
            return self._orders_joined_subset

    def get_products(self):
        return self._products

    def get_aisles(self):
        return self._aisles

    def get_departments(self):
        return self._departments

    def set_subset(self, order_ids):
        self._orders_tip_subset = self._orders_tip[self._orders_tip['order_id'].isin(order_ids)].copy()
        self._orders_joined_subset = self._orders_joined[self._orders_joined['order_id'].isin(order_ids)].copy()
        self._compute_dynamic_features()

    def register_feature(self, feature):
        if isinstance(feature, StaticFeature):
            static_features = [feature.get_feature_name() for feature in self.static_features]
            if feature.get_feature_name() not in static_features:
                self.static_features.add(feature)
        elif isinstance(feature, DynamicFeature):
            dynamic_features = [feature.get_feature_name() for feature in self.dynamic_features]
            if feature.get_feature_name() not in dynamic_features:
                self.dynamic_features.add(feature)

    def unregister_feature(self, feature):
        if feature is StaticFeature:
            self.static_features.remove(feature)
        elif feature is DynamicFeature:
            self.dynamic_features.remove(feature)

    def compute_features(self, only_static=False):
        self._compute_static_features()
        self._merge_static_features_into_subset()
        if not only_static:
            self._compute_dynamic_features()

    def _compute_static_features(self):
        for feature in self.static_features:
            feature.set_orders_tip(self._orders_tip)
            feature.set_orders_joined(self._orders_joined)

            name = feature.get_feature_name()
            if name in self._orders_tip.columns:
                self._orders_tip.drop(name, axis=1, inplace=True)

            feature.compute_feature()

            self._orders_tip = feature.get_orders_tip()
            self._orders_joined = feature.get_orders_joined()

    def _compute_dynamic_features(self):
        for feature in self.dynamic_features:
            feature.set_orders_tip(self._orders_tip_subset)
            feature.set_orders_joined(self._orders_joined_subset)

            name = feature.get_feature_name()
            if name in self._orders_tip_subset.columns:
                self._orders_tip_subset.drop(name, axis=1, inplace=True)

            feature.compute_feature()

            self._orders_tip_subset = feature.get_orders_tip()
            self._orders_joined_subset = feature.get_orders_joined()

    def _merge_static_features_into_subset(self):
        static_features = [feature.get_feature_name() for feature in self.static_features]
        self._orders_tip_subset = self._orders_tip_subset.drop(static_features, axis=1, errors='ignore')
        self._orders_tip_subset = pd.merge(self._orders_tip_subset, self._orders_tip[['order_id'] + static_features],
                                           on='order_id', how='left')
