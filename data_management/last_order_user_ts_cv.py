from sklearn.base import BaseEstimator, TransformerMixin


class LastOrderUserTSCVSplitter(BaseEstimator, TransformerMixin):

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
        validation_sets = self.orders_by_user.groupby('user_id').apply(self._assign_cv_validation_set).reset_index(
            drop=True)
        for i in range(1, self.n_splits + 1):
            train = X.index[(validation_sets['cv_validation_set'] > i)].tolist()
            test = X.index[validation_sets['cv_validation_set'] == i].tolist()

            yield train, test

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
