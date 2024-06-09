from sklearn.base import BaseEstimator, TransformerMixin


class LastOrderUserTSCVSplitter(BaseEstimator, TransformerMixin):

    def __init__(self, data_manager, n_splits):
        self.data_manager = data_manager
        self.n_splits = n_splits
        self.validation_sets = self._assign_cv_validation_set()

    def _assign_cv_validation_set(self):
        """
        Assigns each point of the given user time-series to its respective validation set for cross-validation.
        The inverse column 'order_number' is assigned as validation set for cross-validation.

        :param data_manager: The data manager instance containing the orders time-series.
        :return: The user time-series along with a label 'cv_validation_set' indicating to which validation set
                 the respective point belongs to for cross-validation.
        """
        validation_sets = self.data_manager.get_orders_tip_train()[['user_id', 'order_number']].copy()
        validation_sets['cv_validation_set'] = (validation_sets
                                                .groupby('user_id')
                                                .rank(method='first', ascending=False))

        return validation_sets

    def split(self, X, y=None, groups=None):
        for i in range(1, self.n_splits + 1):
            train = X.index[(self.validation_sets['cv_validation_set'] > i)].tolist()
            test = X.index[self.validation_sets['cv_validation_set'] == i].tolist()

            print(f'Iteration {i}: Train size: {len(train)}, Test size: {len(test)}')
            yield train, test

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
