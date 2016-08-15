import numpy as np
import pandas as pd
import six

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer


def null_filter(x):
    return True


def unflatten_list(llist):
    return [l for sublist in llist for l in sublist]


def dtype_dict(dataframe, dtype_filter=None):
    column_series = dataframe.columns.to_series()
    dtype_groups = column_series.groupby(dataframe.dtypes).groups

    if dtype_filter is None:
        dtype_filter = null_filter

    return {k.name: v for k, v in dtype_groups.items() if dtype_filter(k.name)}


def is_numeric(dtype):
    if isinstance(dtype, six.string_types):
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            return False

    return np.issubdtype(dtype, np.number)


def is_categorical(dtype):
    return not is_numeric(dtype)


def numeric_columns(dataframe):
    return unflatten_list(dtype_dict(dataframe, is_numeric).values())


def categorical_columns(dataframe):
    return unflatten_list(dtype_dict(dataframe, is_categorical).values())


def mean_impute_numerics(dataframe):
    numerics = dataframe.values

    # impute with mean
    imputer = Imputer(strategy='mean', missing_values='NaN')
    numerics = imputer.fit_transform(numerics)

    # apply standard scalar
    scaler = StandardScaler()
    numerics = scaler.fit_transform(numerics)

    return numerics


def replace_null(dataframe, value='NaN', inplace=False):
    dataframe.fillna(value, inplace=inplace)
    return dataframe
