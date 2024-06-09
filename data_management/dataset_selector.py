from sklearn.base import BaseEstimator, TransformerMixin


class DatasetSelector(BaseEstimator, TransformerMixin):
    def __init__(self, prepared_splits_dict):
        self.prepared_splits_dict = prepared_splits_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        hashed_indices = hash(tuple(X.index))
        if hashed_indices not in self.prepared_splits_dict:
            raise ValueError(f'Training/validation dataset with shape {X.shape} not available')

        Xt = self.prepared_splits_dict[hashed_indices]
        if not X.columns.equals(Xt.columns):
            raise ValueError(f'Column mismatch: {X.columns} != {Xt.columns}')

        if X.shape != Xt.shape:
            raise ValueError(f'Shape mismatch: {X.shape} != {Xt.shape}')

        print(f'Transformation: {X.shape} -> {Xt.shape}')
        return Xt
