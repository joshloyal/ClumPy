from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import os

import numpy as np
import pandas as pd

from clumpy.datasets import utils as data_utils
from clumpy.datasets import data_view as dv


root = os.path.abspath(os.path.dirname(__file__))
DATA_NAME = 'data'
FILE_NAME = os.path.join(root, DATA_NAME, '10k_diabetes.csv')
EMBEDDING_FILE_NAME = os.path.join(root, DATA_NAME, '10k_tsne.npy')


class DiabetesView(dv.DataView):
    def as_raw(self):
        try:
            self._data = pd.read_csv(FILE_NAME)
        except IOError:
            raise IOError('Please download 10kdiabetes')

        return self._data

    def as_cleaned(self):
        if self._cleaned is not None:
            return self._cleaned

        df = self.as_raw().copy()

        # remove text columns
        df = df.drop(['readmitted', 'diag_1_desc', 'diag_2_desc', 'diag_3_desc'], axis=1)

        # drop low info
        good_cols = df.apply(lambda x: np.unique(x).shape[0] > 1, axis=0)
        df = df[ good_cols[good_cols].index ]
        self._cleaned = df

        return self._cleaned

    def as_numeric(self):
        df = self.as_cleaned().copy()
        numeric_cols = data_utils.numeric_columns(df)
        return df[numeric_cols]

    def as_nominal(self):
        numeric = self.as_numeric()
        num_x = data_utils.min_max_scale(numeric)
        return pd.DataFrame(num_x, columns=numeric.columns)

    def as_ordinal(self):
        if self._ordinal is not None:
            return self._ordinal

        df = self.as_cleaned().copy()
        #numeric_cols = data_utils.numeric_columns(df)
        #if numeric_cols:
        #    #num_x = data_utils.mean_impute_numerics(df[numeric_cols])
        #    num_x = data_utils.min_max_scale(df[numeric_cols])
        #    numerics = pd.DataFrame(num_x, columns=df[numeric_cols].columns)

        # replace NaN
        categorical_cols = data_utils.categorical_columns(df)
        categorical_df = data_utils.replace_null(df[categorical_cols], value='NaN', inplace=True)
        categoricals = data_utils.label_encode(categorical_df)

        self._ordinal = categoricals
        #self._ordinal = pd.concat([numerics, categoricals], axis=1)

        return self._ordinal

    def as_onehot(self):
        if self._onehot is not None:
            return self._onehot

        df = self.as_cleaned().copy()
        #numeric_cols = data_utils.numeric_columns(df)
        #if numeric_cols:
        #    #num_x = data_utils.mean_impute_numerics(df[numeric_cols])
        #    num_x = data_utils.min_max_scale(df[numeric_cols])
        #    numerics = pd.DataFrame(num_x, columns=df[numeric_cols].columns)


        # replace NaN
        categorical_cols = data_utils.categorical_columns(df)
        categorical_df = data_utils.replace_null(df[categorical_cols], value='NaN', inplace=True)
        categoricals = pd.get_dummies(categorical_df, drop_first=True)

        self._onehot = categoricals#pd.concat([numerics, categoricals], axis=1)

        return self._onehot


def fetch_10kdiabetes():
    return DiabetesView()


def fetch_10kdiabetes_embedding():
    data = np.load(EMBEDDING_FILE_NAME)
    return data
