import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d, check_array
from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler
import pandas as pd


def column_atleast_2d(array):
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def fit_encode_1d(y, strategy='frequency'):
    y = column_or_1d(y, warn=True)
    levels = np.unique(y)

    if strategy == 'frequency':
        frequencies = [np.sum(y == level) for level in levels]
        levels = levels[np.argsort(frequencies)]

    return levels


def transform_encode_1d(y, fit_levels):
    levels = np.unique(y)
    if len(np.intersect1d(levels, fit_levels)) < len(levels):
        diff = np.setdiff1d(levels, fit_levels)
        raise ValueError("y contains new labels: %s" % str(diff))
    return np.searchsorted(fit_levels, y).reshape(-1, 1)



def inverse_encode_1d(y, fit_levels):
    """There is probably a built in numpy method for this..."""
    vec_map = np.vectorize(lambda x: fit_levels[x])
    return vec_map(y)


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='frequency'):
        self.strategy = strategy
        self.level_map = {}

    def fit(self, X):
        X = column_atleast_2d(X)

        self.n_features_ = X.shape[1]
        self.level_map = {column_idx: fit_encode_1d(X[:, column_idx]) for
                          column_idx in xrange(self.n_features_)}

        return self

    def transform(self, X):
        X = column_atleast_2d(X)
        if X.shape[1] != self.n_features_:
            raise ValueError("Different number of features at transform time.",
                             " n_features_transform= %d" % X.shape[1],
                             " and n_features_fit= %d" % self.n_features_)

        return np.hstack([transform_encode_1d(X[:, column_idx], levels) for
                          column_idx, levels in self.level_map.iteritems()])

    def inverse_transform(self, X):
        X = column_atleast_2d(X)
        if X.shape[1] != self.n_features_:
            raise ValueError("Different number of features at transform time.",
                             " n_features_transform= %d" % X.shape[1],
                             " and n_features_fit= %d" % self.n_features_)

        return np.hstack([inverse_encode_1d(X[:, column_idx], levels) for
                         column_idx, levels in self.level_map.iteritems()])


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
            dummy_data = pd.get_dummies(X[categorical_columns], columns=categorical_columns, dummy_na=True)
            categorical_columns = dummy_data.columns.tolist()
            cat_X = dummy_data.values

        if num_columns:
            num_X = imputer.fit_transform(X[num_columns].values)
            if scaler:
                num_X = scaler.fit_transform(num_X)

            return np.hstack((num_X, cat_X)), num_columns, categorical_columns

        return cat_X, [], categorical_columns

    else:
        num_X = imputer.fit_transform(X.values)
        if scaler:
            num_X = scaler.fit_transform(num_X)

        return num_X, X.columns.tolist(), []
