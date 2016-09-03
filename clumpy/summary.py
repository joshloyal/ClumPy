from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.utils import check_random_state

from clumpy.datasets import utils as data_utils
from clumpy.cluster_rules import ova_forest_importance

@np.vectorize
def as_factors(x):
    factor_map = {1: 'LOW', 2: 'MEDIUM', 3: 'HIGH'}
    return factor_map.get(x, 'UNK')


def mode_aggregate(x):
    return stats.mode(x)[0].item()


def bin_numeric_column(X, bins=10, random_state=1234):
    X = X.values
    n_samples = X.shape[0]

    rng = check_random_state(random_state)
    X = X + rng.rand(n_samples) * 1e-6

    percentiles = np.arange(1, bins-1) * 1. / bins
    breaks = stats.mstats.mquantiles(X, np.hstack((0, percentiles, 1)))
    X_binned = np.digitize(X, breaks)

    #return as_factors(X_binned)
    return X_binned



def cluster_summary(df, cluster_labels):
    data = df.copy()

    # calculate overall statistics
    stats = data.median()

    #groupby cluster
    data['cluster_id'] = cluster_labels
    #numeric_cols = data_utils.numeric_columns(data)
    #categorical_cols = data_utils.categorical_columns(data)
    #data['cluster'] = clusterer.labels_

    #if bin_numeric:
    #    data[numeric_cols] = data[numeric_cols].apply(bin_numeric_column, axis=1)
    #    numeric_summary = data[
    #        numeric_cols + ['cluster']].groupby('cluster').agg(
    #                mode_aggregate)
    #else:
    #    numeric_summary = data[numeric_cols + ['cluster']].groupby('cluster').median()

    ## use modes for categoricals
    #categorical_summary = data[
    #        categorical_cols + ['cluster']].groupby('cluster').agg(
    #                mode_aggregate)

    #return pd.concat([numeric_summary, categorical_summary], axis=1)

    group_stats = data.groupby('cluster_id').median()
    group_stats.loc['overall'] = stats
    return group_stats


def flat_bar(frame, feature_name, class_column):
    import matplotlib.pyplot as plt

    n_samples = len(frame)
    classes = frame[class_column].drop_duplicates()
    class_col = frame[class_column]
    df = frame[feature_name]

    ax = plt.gca()
    for sample_idx in range(n_samples):
        y = df.iloc[sample_idx].values


