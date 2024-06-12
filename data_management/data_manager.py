import zipfile
from pathlib import Path

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

    def get_orders_tip_train(self, complete=False):
        if complete:
            return self._orders_tip[self._orders_tip['tip'].notnull()].reset_index(drop=True)
        else:
            return self._orders_tip_subset[self._orders_tip_subset['tip'].notnull()].reset_index(drop=True)

    # TODO: Adjust to retrieve the correct test set
    def get_orders_tip_test(self, complete=False):
        if complete:
            return self._orders_tip[self._orders_tip['tip'].isnull()]
        else:
            return self._orders_tip_subset[self._orders_tip_subset['tip'].isnull()]

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

    def set_subset(self, order_ids, reset_index=True, add_remove_first_orders=False):
        indices = order_ids.sort_index().index
        if add_remove_first_orders:
            first_order_ids = self._orders_tip[self._orders_tip['order_number'] == 1]['order_id']
            order_ids = pd.concat([order_ids, first_order_ids]).drop_duplicates(keep=False)

        order_ids = order_ids.sort_index()
        self._orders_tip_subset = pd.merge(self._orders_tip, order_ids, how='inner')
        self._orders_joined_subset = pd.merge(self._orders_joined, order_ids, how='inner')
        self._compute_dynamic_features()

        if add_remove_first_orders:
            self._remove_first_orders()

        if reset_index:
            self._reset_index()
        else:
            self._orders_tip_subset = self._orders_tip_subset.set_index(indices)

    def remove_first_orders(self):
        self._remove_first_orders()
        self._reset_index()

    def _reset_index(self):
        self._orders_tip_subset = self._orders_tip_subset.reset_index(drop=True)
        self._orders_joined_subset = self._orders_joined_subset.reset_index(drop=True)

    def _remove_first_orders(self):
        self._orders_tip_subset = self._orders_tip_subset[self._orders_tip_subset['order_number'] > 1]
        self._orders_joined_subset = self._orders_joined_subset[self._orders_joined_subset['order_number'] > 1]

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

    def get_registered_features(self):
        return [feature.get_feature_name() for feature in self.static_features.union(self.dynamic_features)]

    def get_registered_static_features(self):
        return [feature.get_feature_name() for feature in self.static_features]

    def get_registered_dynamic_features(self):
        return [feature.get_feature_name() for feature in self.dynamic_features]

    def compute_features(self, only_static=False):
        self._compute_static_features()
        self._merge_static_features_into_subset()
        if not only_static:
            self._compute_dynamic_features()

    def _compute_static_features(self):
        original_index = self._orders_tip_subset.index
        original_index_joined = self._orders_joined_subset.index

        for feature in self.static_features:
            feature.set_orders_tip(self._orders_tip)
            feature.set_orders_joined(self._orders_joined)

            name = feature.get_feature_name()
            if name in self._orders_tip.columns:
                self._orders_tip.drop(name, axis=1, inplace=True)

            feature.compute_feature()

            self._orders_tip = feature.get_orders_tip()
            self._orders_joined = feature.get_orders_joined()

        self._orders_tip.index = original_index
        self._orders_joined.index = original_index_joined

    def _compute_dynamic_features(self):

        original_index = self._orders_tip_subset.index
        original_index_joined = self._orders_joined_subset.index

        for feature in self.dynamic_features:
            feature.set_orders_tip(self._orders_tip_subset)
            feature.set_orders_joined(self._orders_joined_subset)

            name = feature.get_feature_name()
            if name in self._orders_tip_subset.columns:
                self._orders_tip_subset.drop(name, axis=1, inplace=True)

            feature.compute_feature()

            self._orders_tip_subset = feature.get_orders_tip()
            self._orders_joined_subset = feature.get_orders_joined()

        self._orders_tip_subset.index = original_index
        self._orders_joined_subset.index = original_index_joined

    def _merge_static_features_into_subset(self):
        static_features = [feature.get_feature_name() for feature in self.static_features]
        self._orders_tip_subset = self._orders_tip_subset.drop(static_features, axis=1, errors='ignore')
        self._orders_tip_subset = pd.merge(self._orders_tip_subset, self._orders_tip[['order_id'] + static_features],
                                           on='order_id', how='left')

    def export_features(self, path, only_static=False):
        features = [feature.get_feature_name() for feature in self.static_features]
        if not only_static:
            features += [feature.get_feature_name() for feature in self.dynamic_features]
            if not self._orders_tip_subset['order_id'].equals(self._orders_tip['order_id']):
                raise ValueError('Dynamic features have not been computed for the complete dataset.')
            else:
                df_to_export = self._orders_tip_subset
        else:
            df_to_export = self._orders_tip

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(filepath, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
            csv_filename = filepath.stem + '.csv'
            with archive.open(csv_filename, mode='w') as csv_file:
                df_to_export.to_csv(csv_file, columns=['order_id'] + features, index=False)

    def import_features(self, path, only_static=False):
        filepath = Path(path)
        orders_tip_imported = pd.read_csv(filepath)

        static_features = [feature.get_feature_name() for feature in self.static_features]
        columns_to_merge_static = [col for col in static_features if col in orders_tip_imported.columns] + ['order_id']
        self._orders_tip = self._orders_tip.drop(static_features, axis=1, errors='ignore')
        self._orders_tip = pd.merge(self._orders_tip, orders_tip_imported[columns_to_merge_static], on='order_id',
                                    how='left')
        self._merge_static_features_into_subset()

        if not only_static:
            dynamic_features = [feature.get_feature_name() for feature in self.dynamic_features]
            columns_to_merge_dynamic = [col for col in dynamic_features if col in orders_tip_imported.columns] + [
                'order_id']
            self._orders_tip_subset = self._orders_tip_subset.drop(dynamic_features, axis=1, errors='ignore')
            self._orders_tip_subset = pd.merge(self._orders_tip_subset, orders_tip_imported[columns_to_merge_dynamic],
                                               on='order_id', how='left')
