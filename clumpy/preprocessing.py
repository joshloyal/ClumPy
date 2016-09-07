import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler
import pandas as pd


class _OrdinalEncoder1D(BaseEstimator, TransformerMixin):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.levels_ = np.unique(y)
        self.frequencies_ = [np.sum(y == level) for
                             level in self.levels_]
        self.levels_ = self.levels_[np.argsort(self.frequencies_)]

        return self

    def transform(self, y):
        levels = np.unique(y)
        if len(np.intersect1d(levels, self.levels_)) < len(levels):
            diff = np.setdiff1d(levels, self.levels_)
            raise ValueError("y contains new labels: %s" % str(diff))
        return np.searchsorted(self.levels_, y)


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X):
        self.encoders = [
                _OrdinalEncoder1D().fit(X[:, col_id]) for
                col_id in xrange(X.shape[1])]
        return self

    def transform(self, X):
        return np.hstack([
            encoder.transform(X[:, col_id])[:, np.newaxis] for
            col_id, encoder in enumerate(self.encoders)])


class ArbitraryImputer(BaseEstimator, TransformerMixin):
    def __init__(self, impute_value):
        self.impute_value = impute_value

    def fit(self, X):
        return self

    def transform(self, X):
        mask = np.isfinite(X)
        if ~np.all(mask):
            np.putmask(X, ~mask, self.impute_value)
        return X


def process_data(X, categorical_columns=[], impute='arbitrary', cat_preprocessing='ordinal', num_preprocessing=None):
    if impute == 'mean':
        imputer = Imputer(strategy='mean', missing_values='NaN')
    else:
        imputer = ArbitraryImputer(impute_value=-9999)

    if num_preprocessing == 'standardize':
        scaler = StandardScaler()
    elif num_preprocessing == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    if categorical_columns:
        num_columns = [col for col in X.columns if
                       col not in categorical_columns]


        if cat_preprocessing == 'ordinal':
            cat_X = OrdinalEncoder().fit_transform(X[categorical_columns].values)
        else:
            dummy_data = pd.get_dummies(X[categorical_columns], dummy_na=True)
            categorical_columns = dummy_data.columns.tolist()
            cat_X = dummy_data.values

        num_X = imputer.fit_transform(X[num_columns].values)
        if scaler:
            num_X = scaler.fit_transform(num_X)

        return np.hstack((num_X, cat_X)), num_columns, categorical_columns

    else:
        num_X = imputer.fit_transform(X.values)
        if scaler:
            num_X = scaler.fit_transform(num_X)

        return num_X, X.columns.tolist(), []
