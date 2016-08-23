from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import os

import pandas as pd

from clumpy.datasets import utils as data_utils
from clumpy.datasets import data_view as dv


root = os.path.abspath(os.path.dirname(__file__))
DATA_NAME = 'data'
FILE_NAME = os.path.join(root, DATA_NAME, 'cars.csv')


class CarsView(dv.DataView):
    def as_raw(self):
        try:
            self._data = pd.read_csv(FILE_NAME)
        except IOError:
            raise IOError('Please download cars')

        return self._data

    def as_numeric(self):
        df = self.as_raw().copy()
        numeric_cols = data_utils.numeric_columns(df)
        numerics = data_utils.impute_nan(df[numeric_cols])

        return numerics

    def as_ordinal(self):
        if self._ordinal is not None:
            return self._ordinal

        df = self.as_raw().copy()

        # replace NaN
        categorical_cols = data_utils.categorical_columns(df)
        categorical_df = data_utils.replace_null(df[categorical_cols], value='NaN', inplace=True)
        categoricals = data_utils.label_encode(categorical_df)

        self._ordinal = categoricals

        return self._ordinal

def fetch_cars(no_processing=False):
    #df = pd.read_csv(FILE_NAME)
    #numeric_cols = data_utils.numeric_columns(df)

    #if no_processing:
    #    return df[numeric_cols]
    #else:
    #    data = data_utils.mean_impute_numerics(df[numeric_cols])
    #    return pd.DataFrame(data, columns=df[numeric_cols].columns)

    return CarsView()
