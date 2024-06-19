import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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
            return self._orders_tip[self._orders_tip['eval_set'] != 'train'].reset_index(drop=True)
        else:
            return self._orders_tip_subset[self._orders_tip_subset['eval_set'] != 'train'].reset_index(drop=True)

    def get_orders_tip_test(self, complete=False):
        if complete:
            return self._orders_tip[self._orders_tip['eval_set'] == 'train']
        else:
            return self._orders_tip_subset[self._orders_tip_subset['eval_set'] == 'train']

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

    def set_subset(self, order_ids, reset_index=True, add_remove_first_orders=False, set_tips_to_nan=None):
        indices = order_ids.sort_index().index
        if add_remove_first_orders:
            first_order_ids = self._orders_tip[self._orders_tip['order_number'] == 1]['order_id']
            order_ids = pd.concat([order_ids, first_order_ids]).drop_duplicates(keep=False)

        order_ids = order_ids.sort_index()
        self._orders_tip_subset = pd.merge(self._orders_tip, order_ids, how='inner')
        self._orders_joined_subset = pd.merge(self._orders_joined, order_ids, how='inner')

        if set_tips_to_nan is not None:
            self.set_tip_to_nan(set_tips_to_nan)

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

    def set_tip_to_nan(self, order_ids):
        self._orders_tip_subset.loc[self._orders_tip_subset['order_id'].isin(order_ids), 'tip'] = np.nan
        self._orders_joined_subset.loc[self._orders_joined_subset['order_id'].isin(order_ids), 'tip'] = np.nan

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

    def calculate_feature_correlations(self, only_static=False):
        static_features = [feature.get_feature_name() for feature in self.static_features]
        static_features.append('order_number')
        static_features.append('days_since_prior_order')

        if not only_static:
            dynamic_features = [feature.get_feature_name() for feature in self.dynamic_features]
            all_features = static_features + dynamic_features
            orders_tip_features = self._orders_tip_subset[self._orders_tip_subset['order_number'] != 1]
        else:
            all_features = static_features
            orders_tip_features = self._orders_tip[self._orders_tip['order_number'] != 1]

        correlations = {}
        for feature in all_features:
            if feature in orders_tip_features.columns:
                correlations[feature] = orders_tip_features[feature].corr(orders_tip_features['tip'])

        correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])

        correlation_df = correlation_df.sort_values(by='Correlation', key=abs, ascending=False).reset_index(drop=True)
        print(correlation_df)

    def visualize_correlation_between_features(self, only_static=False):
        static_features = [feature.get_feature_name() for feature in self.static_features]
        static_features.append('order_number')
        static_features.append('days_since_prior_order')

        if not only_static:
            dynamic_features = [feature.get_feature_name() for feature in self.dynamic_features]
            orders_tip_features = self._orders_tip_subset[self._orders_tip_subset['order_number'] != 1]
            all_features = static_features + dynamic_features
        else:
            orders_tip_features = self._orders_tip[self._orders_tip['order_number'] != 1]
            all_features = static_features

        # print(all_features)

        filtered_data = orders_tip_features[all_features]
        print(filtered_data.corr())
        # scatter_matrix = pd.plotting.scatter_matrix(filtered_data, figsize=(12, 12), diagonal='kde')

        # plt.show()

    def visualize_feature_analysis(self, only_static=False):
        static_features = [feature.get_feature_name() for feature in self.static_features]
        static_features.append('order_number')
        static_features.append('days_since_prior_order')
        dynamic_features = [feature.get_feature_name() for feature in self.dynamic_features]
        orders_tip_features = self._orders_tip[self._orders_tip['order_number'] != 1]

        print(f'Analyze of static features:')
        self._plot_feature_analysis(static_features, 2, orders_tip_features)

        if not only_static:
            orders_tip_features = self._orders_tip_subset[self._orders_tip_subset['order_number'] != 1]
            print(f'Analyze of dynamic features:')
            self._plot_feature_analysis(dynamic_features, 2, orders_tip_features)

    def _plot_feature_analysis(self, list_of_features, number_of_plots_per_row, orders_tip_features):
        tip = orders_tip_features['tip']
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        for i, feature in enumerate(list_of_features):
            position = i % number_of_plots_per_row
            feature_data = orders_tip_features[feature]
            feature_tip_correlation = feature_data.corr(tip)

            if number_of_plots_per_row == 1:
                sns.violinplot(data=orders_tip_features, x='tip', y=feature, ax=axs)
                axs.set_title(f'{feature} vs tip correlation: {feature_tip_correlation}')
                axs.set_xlabel('tip')
                axs.set_ylabel(feature)
            else:
                sns.violinplot(data=orders_tip_features, x='tip', y=feature, ax=axs[position])
                axs[position].set_title(f'{feature} vs tip correlation: {feature_tip_correlation}')
                axs[position].set_xlabel('tip')
                axs[position].set_ylabel(feature)

            if position == number_of_plots_per_row - 1:
                plt.tight_layout()
                plt.show()
                if i != len(list_of_features) - 1:
                    if len(list_of_features) - 1 - i < number_of_plots_per_row:
                        number_of_plots_per_row = len(list_of_features) - 1 - i
                        fig, axs = plt.subplots(1, number_of_plots_per_row, figsize=(15, 5))
                    else:
                        fig, axs = plt.subplots(1, number_of_plots_per_row, figsize=(15, 5))

    def analyse_each_feature(self, only_static=False):
        print(f'Analyze of static features:')
        for feature in self.static_features:
            orders_tip_features = self._orders_tip[self._orders_tip['order_number'] != 1]
            feature.analyze_feature(orders_tip_features)

        if not only_static:
            print(f'Analyze of dynamic features:')
            for feature in self.dynamic_features:
                orders_tip_features = self._orders_tip_subset[self._orders_tip_subset['order_number'] != 1]
                feature.analyze_feature(orders_tip_features)

