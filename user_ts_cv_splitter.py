import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class UserTSCVSplitter(BaseEstimator, TransformerMixin):

    def __init__(self, n_splits, validation_set_ratio, orders_by_user):
        self.n_splits = n_splits
        self.validation_set_ratio = validation_set_ratio
        self.orders_by_user = orders_by_user

    def _assign_cv_validation_set(self, user_ts):
        """
        Assigns each point of the given user time-series to its respective validation set for cross-validation.
        It is iterated across the given user time-series and the last 'validation_set_ratio'% points of the remaining
        points are assigned to the same validation set for cross-validation.

        :param user_ts: The user time-series that shall be prepared for cross-validation.
        :return: The given user time-series along with a label 'cv_validation_set' indicating to which validation set
                 the respective point belongs to for cross-validation.
        """
        num_remaining_orders = len(user_ts)
        cv_validation_set = 1
        user_ts_temp = user_ts.copy()

        while num_remaining_orders > 0:
            num_orders_to_assign = num_remaining_orders * self.validation_set_ratio
            if not num_orders_to_assign.is_integer():
                num_orders_to_assign += np.random.choice([0, 1], p=[1 - self.validation_set_ratio, self.validation_set_ratio])
            num_orders_to_assign = int(num_orders_to_assign)

            if num_orders_to_assign > 0:
                idx_to_assign = user_ts_temp.index[-num_orders_to_assign:]
                user_ts.loc[idx_to_assign, 'cv_validation_set'] = cv_validation_set
                user_ts_temp = user_ts_temp.iloc[:-num_orders_to_assign]
                num_remaining_orders -= num_orders_to_assign

            cv_validation_set += 1

        user_ts['cv_validation_set'] = user_ts['cv_validation_set'].astype(int)
        return user_ts

    def split(self, X, y=None, groups=None):
        validation_sets = self.orders_by_user.groupby('user_id').apply(self._assign_cv_validation_set).reset_index(drop=True)
        for i in range(1, self.n_splits + 1):
            train = X.index[validation_sets['cv_validation_set'] > i].tolist()
            test = X.index[validation_sets['cv_validation_set'] == i].tolist()
            yield train, test
                   
    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class LastUserTSCVSplitter(BaseEstimator, TransformerMixin):

    def __init__(self, n_splits, orders_by_user):
        self.n_splits = n_splits
        self.orders_by_user = orders_by_user

    def _assign_cv_validation_set(self, user_ts):
        """
        Assigns each point of the given user time-series to its respective validation set for cross-validation.
        The inverse column 'order_number' is assigned as validation set for cross-validation.

        :param user_ts: The user time-series that shall be prepared for cross-validation.
        :return: The given user time-series along with a label 'cv_validation_set' indicating to which validation set
                 the respective point belongs to for cross-validation.
        """
        user_ts['cv_validation_set'] = user_ts['order_number'].values[::-1]
        return user_ts

    def split(self, X, y=None, groups=None):
        validation_sets = self.orders_by_user.groupby('user_id').apply(self._assign_cv_validation_set).reset_index(drop=True)
        for i in range(1, self.n_splits + 1):
            train = X.index[(validation_sets['cv_validation_set'] > i)].tolist()
            test = X.index[validation_sets['cv_validation_set'] == i].tolist()
            yield train, test

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
