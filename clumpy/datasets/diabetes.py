from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import os

import numpy as np
import pandas as pd

from clumpy.datasets import utils as data_utils


root = os.path.abspath(os.path.dirname(__file__))
DATA_NAME = 'data'
FILE_NAME = os.path.join(root, DATA_NAME, '10k_diabetes.csv')
EMBEDDING_FILE_NAME = os.path.join(root, DATA_NAME, '10k_tsne.npy')


def fetch_10kdiabetes(only_numerics=False, only_categoricals=False, one_hot=False):
    # load data
    try:
        df = pd.read_csv(FILE_NAME)
    except IOError:
        raise IOError('Please download 10kdiabetes')

    # don't use the target for clustering for now
    df.pop('readmitted')

    # remove text columns
    df = df.drop(['diag_1_desc', 'diag_2_desc', 'diag_3_desc'], axis=1)

    # drop low info
    good_cols = df.apply(lambda x: np.unique(x).shape[0] > 1, axis=0)
    df = df[ good_cols[good_cols].index ]

    # call before replace None...
    numeric_cols = data_utils.numeric_columns(df)
    if numeric_cols:
        num_x = data_utils.mean_impute_numerics(df[numeric_cols])
        numerics = pd.DataFrame(num_x, columns=df[numeric_cols].columns)


    # replace NaN
    categorical_cols = data_utils.categorical_columns(df)
    categorical_df = data_utils.replace_null(df[categorical_cols], value='NaN', inplace=True)
    categoricals = pd.get_dummies(categorical_df, drop_first=True)

    if only_numerics:
        return numerics
    elif only_categoricals:
        return categoricals
    else:
        return pd.concat([numerics, categoricals], axis=1)


def fetch_10kdiabetes_embedding():
    data = np.load(EMBEDDING_FILE_NAME)
    return data
