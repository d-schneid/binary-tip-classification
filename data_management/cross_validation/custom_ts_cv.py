import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class CustomTSCVSplitter(ABC):
    def __init__(self, data_manager, n_splits, splitter):
        self.data_manager = data_manager
        self.n_splits = n_splits
        self.splitter = splitter

    @abstractmethod
    def split(self, X, y=None, groups=None):
        pass

    @abstractmethod
    def get_n_splits(self, X, y, groups):
        pass

    def export_splits(self, path):
        dir = Path(path)
        dir.mkdir(parents=True, exist_ok=True)
        all_orders_tip_train = self.data_manager.get_orders_tip_train().copy()

        for i, (train, test) in enumerate(self.split(all_orders_tip_train)):
            order_ids_test = all_orders_tip_train.loc[test]['order_id']
            order_ids = all_orders_tip_train.loc[test + train]['order_id']
            self.data_manager.set_subset(order_ids, reset_index=False, add_remove_first_orders=True,
                                         set_tips_to_nan=order_ids_test)
            orders_tip_train = self.data_manager.get_orders_tip()

            train_filename = dir / f'{self.splitter}_train_{i + 1}.csv.zip'
            test_filename = dir / f'{self.splitter}_test_{i + 1}.csv.zip'

            self._save_df_to_csv(orders_tip_train.loc[train], train_filename)
            self._save_df_to_csv(orders_tip_train.loc[test], test_filename)

            print(f'Export {i + 1}: Train size: {len(train)}, Test size: {len(test)}')

        order_ids_test = self.data_manager.get_orders_tip_test(complete=True)['order_id']
        order_ids_train = all_orders_tip_train['order_id']

        all_order_ids = pd.concat([order_ids_train, order_ids_test])
        self.data_manager.set_subset(all_order_ids, add_remove_first_orders=True)
        all_orders_tip_train = self.data_manager.get_orders_tip_train()
        all_orders_tip_test = self.data_manager.get_orders_tip_test()

        all_filename_train = dir / f'{self.splitter}_all_train.csv.zip'
        all_filename_test = dir / f'{self.splitter}_all_test.csv.zip'

        self._save_df_to_csv(all_orders_tip_train, all_filename_train)
        self._save_df_to_csv(all_orders_tip_test, all_filename_test)

    def _save_df_to_csv(self, df_to_export, filepath):
        with zipfile.ZipFile(filepath, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
            csv_filename = filepath.stem + '.csv'
            with archive.open(csv_filename, mode='w') as csv_file:
                df_to_export.to_csv(csv_file, index=True)

    def import_splits(self, path, features):
        dir = Path(path)
        splits = {}

        for i in range(self.n_splits):
            train_filename = dir / f'{self.splitter}_train_{i + 1}.csv.zip'
            test_filename = dir / f'{self.splitter}_test_{i + 1}.csv.zip'

            train = self._load_df_from_csv(train_filename, features)
            test = self._load_df_from_csv(test_filename, features)

            # Use the hashed indices of the dataframes as keys
            splits[hash(tuple(train.index))] = train
            splits[hash(tuple(test.index))] = test

        all_filename_train = dir / f'{self.splitter}_all_train.csv.zip'
        all_filename_test = dir / f'{self.splitter}_all_test.csv.zip'

        all_df_train = self._load_df_from_csv(all_filename_train, features)
        all_df_test = self._load_df_from_csv(all_filename_test, features)

        splits[hash(tuple(all_df_train.index))] = all_df_train
        splits[hash(tuple(all_df_test.index))] = all_df_test

        return splits

    def _load_df_from_csv(self, filename, features):
        with zipfile.ZipFile(filename, mode='r') as archive:
            csv_filename = filename.stem + '.csv'
            with archive.open(csv_filename, mode='r') as csv_file:
                df = pd.read_csv(csv_file, index_col=0)

        return df[features]
