from __future__ import division

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
import pandas as pd


from clumpy.unique_threshold import UniqueThreshold
from clumpy.datasets.utils import ordinal_columns, continous_columns, categorical_columns


def column_atleast_2d(array):
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def fit_encode_1d(y, strategy='frequency'):
    y = column_atleast_2d(y)
    levels = np.unique(y)

    # FIXME: serach sorted doesn't work here...
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
    vec_map = np.vectorize(lambda x: fit_levels[int(x)])
    return vec_map(y).reshape(-1, 1)


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='random'):
        self.strategy = strategy
        self.level_map = []

    def fit(self, X):
        X = column_atleast_2d(X)

        self.n_features_ = X.shape[1]
        self.level_map = [
            fit_encode_1d(X[:, column_idx], strategy=self.strategy) for
            column_idx in xrange(self.n_features_)]

        return self

    def transform(self, X):
        X = column_atleast_2d(X)
        if X.shape[1] != self.n_features_:
            raise ValueError("Different number of features at transform time.",
                             " n_features_transform= %d" % X.shape[1],
                             " and n_features_fit= %d" % self.n_features_)

        return np.hstack([transform_encode_1d(X[:, column_idx], levels) for
                          column_idx, levels in enumerate(self.level_map)])

    def inverse_transform(self, X):
        X = column_atleast_2d(X)
        if X.shape[1] != self.n_features_:
            raise ValueError("Different number of features at transform time.",
                             " n_features_transform= %d" % X.shape[1],
                             " and n_features_fit= %d" % self.n_features_)

        encoding = np.hstack([inverse_encode_1d(X[:, column_idx], levels) for
                          column_idx, levels in enumerate(self.level_map)])
        return encoding


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


def median_impute(X, strategy='median'):
    imputer = Imputer(strategy=strategy, missing_values='NaN')
    return imputer.fit_transform(X)


def scale_values(X, strategy='standardize'):
    if strategy == 'standardize':
        scaler = StandardScaler()
    elif strategy == 'center':
        scaler = StandardScaler(with_mean=True, with_std=False)
    elif strategy == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError('Unrecognized scaling strategy `%s`.' % strategy)

    return scaler.fit_transform(X)

def remove_low_variance(X, threshold=0.0):
    """Remove columns with low variance."""
    selector = VarianceThreshold(threshold=threshold)
    return selector.fit_transform(X)


def remove_low_info(X, max_frequency=0.99):
    """remove columns that have too much variance (a lot of unique values)"""
    selector = UniqueThreshold(max_frequency=max_frequency)
    return selector.fit_transform(X)


def encode_values(X, strategy='onehot'):
    if strategy == 'onehot':
        return pd.get_dummies(X, dummy_na=True).values
    elif strategy == 'none':
        return X.values
    else:
        raise ValueError('Unrecognized encoding strategy `%s`.' % strategy)


def process_continous(X):
    """Continous numeric value preprocessing."""
    # missing value imputation
    X = median_impute(X, strategy='median')

    # remove low variance variables
    X = remove_low_variance(X)

    # scaling
    X = scale_values(X, strategy='standardize')

    return X.astype(np.float64)


def process_ordinal(X):
    """ordinal numeric value preprocessing."""
    # missing value imputation
    X = median_impute(X, strategy='median')

    # remove any low info columns (high variance)
    X = remove_low_info(X)

    # remove low variance variables
    X = remove_low_variance(X)

    # scaling
    X = scale_values(X, strategy='standardize')

    return X.astype(np.float64)


def process_categorical(X):
    # encode categoricals as numeric
    X = encode_values(X, strategy='onehot')

    # remove any low info columns
    X = remove_low_info(X)

    # remove low variance variables
    X = remove_low_variance(X)

    # scaling
    #X = scale_values(X, strategy='center')

    return X.astype(np.float64)


def process_data(df):
    # categorize columns
    categorical_cols = categorical_columns(df)
    ordinal_cols = ordinal_columns(df)
    continous_cols = continous_columns(df)

    # pre-process
    continous_X, ordinal_X, categorical_X = None, None, None
    if categorical_cols:
        categorical_X = process_categorical(df[categorical_cols])
    if ordinal_cols:
        ordinal_X = process_ordinal(df[ordinal_cols].values)
    if continous_cols:
        continous_X = process_continous(df[continous_cols].values)


    return continous_X, ordinal_X, categorical_X
    #if num_preprocessing == 'standardize':
    #    scaler = StandardScaler()
    #elif num_preprocessing == 'minmax':
    #    scaler = MinMaxScaler()
    #else:
    #    scaler = None

    #if categorical_columns:
    #    num_columns = [col for col in X.columns if
    #                   col not in categorical_columns]


    #    if cat_preprocessing == 'ordinal':
    #        cat_X = OrdinalEncoder().fit_transform(X[categorical_columns].values)
    #    else:
    #        dummy_data = pd.get_dummies(X[categorical_columns], columns=categorical_columns, dummy_na=True)
    #        categorical_columns = dummy_data.columns.tolist()
    #        cat_X = dummy_data.values

    #    if num_columns:
    #        num_X = imputer.fit_transform(X[num_columns].values)
    #        if scaler:
    #            num_X = scaler.fit_transform(num_X)

    #        return np.hstack((num_X, cat_X)), num_columns, categorical_columns

    #    return cat_X, [], categorical_columns

    #else:
    #    num_X = imputer.fit_transform(X.values)
    #    if scaler:
    #        num_X = scaler.fit_transform(num_X)

    #    return num_X, X.columns.tolist(), []
