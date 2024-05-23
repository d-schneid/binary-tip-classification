import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Feature(ABC):

    def __init__(self, op_prior, op_train, tip_train, tip_test, orders):
        self.op_prior = op_prior
        self.op_train = op_train
        self.op = pd.concat([self.op_prior, self.op_train])

        self.tip_train = tip_train
        self.tip_test = tip_test
        self.tip = pd.concat([self.tip_train, self.tip_test[['order_id', 'tip']]])

        self.orders = orders
        self.orders_tip = pd.merge(self.orders, self.tip)

    def compute_feature(self):
        self._handle_missing_values()
        self._compute_feature()

    def _handle_missing_values(self):
        self.orders_tip['days_since_prior_order'] = self.orders_tip['days_since_prior_order'].fillna(-1).astype(int)

    @abstractmethod
    def _compute_feature(self):
        pass

    def analyze_feature(self):
        pass

    @abstractmethod
    def _analyze_feature(self):
        pass


class TipHistory(Feature):

    def __init__(self, op_prior, op_train, tip_train, tip_test, orders):
        super().__init__(op_prior, op_train, tip_train, tip_test, orders)

    def _compute_feature(self):
        self.orders_tip['tip'] = self.orders_tip['tip'].astype(bool)
        self.orders_tip['tip_history'] = (self.orders_tip.groupby('user_id')['tip'].transform('cumsum').shift(1) /
                                          self.orders_tip['order_number'].shift(1))

        self.orders_tip.loc[self.orders_tip['order_number'] == 1, 'tip_history'] = -1

        self.orders_tip['tip'] = self.orders_tip['tip'].astype(object)
        self.orders_tip.loc[self.orders_tip['order_id'].isin(self.tip_test['order_id']), 'tip'] = np.nan

    def _analyze_feature(self):
        pass
