import time

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.utils import check_random_state
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

from kmodes import kmodes
import clumpy.cluster.k_modes as k_modes
from clumpy.preprocessing import OrdinalEncoder
from clumpy.cluster.tests import soybean_data


def gen_data(n_samples=100, random_state=1):
    random_state = check_random_state(random_state)

    x1 = random_state.choice(np.arange(2), n_samples, p=[0.2, 0.8]).reshape(-1, 1)
    x2 = random_state.choice(np.arange(10), n_samples).reshape(-1, 1)
    x3 = random_state.choice(np.arange(3), n_samples, p=[0.1, 0.2, 0.7]).reshape(-1, 1)

    return np.hstack((x1, x2, x3))


def test_cao_density():
    X = np.array([
        [1, 0, 1],
        [3, 2, 0],
        [1, 2, 0],
    ])
    encoder = OrdinalEncoder(strategy='none')
    ordinal_X = encoder.fit_transform(X)
    density = k_modes._categorical_density(ordinal_X)
    np.testing.assert_array_equal(density, np.array([4, 5, 6]))


def test_cao_init(soybean_data):
    X = np.array([
        [1, 0, 1],
        [3, 2, 0],
        [1, 2, 0],
    ])
    encoder = OrdinalEncoder(strategy='none')
    ordinal_X = encoder.fit_transform(X)
    centers = encoder.inverse_transform(
            k_modes._cao_init(ordinal_X, 3))

    expected = np.array([
        [1, 2, 0],
        [1, 0, 1],
        [3, 2, 0]])
    np.testing.assert_array_equal(centers, expected)


def test_labels_inertia(soybean_data):
    X = np.array([
        [1, 0, 1],
        [3, 2, 0],
        [1, 2, 0],
        [1, 0, 3],
    ])

    encoder = OrdinalEncoder(strategy='none')
    ordinal_X = encoder.fit_transform(X)
    centers = ordinal_X[:3]
    labels, inertia = k_modes._labels_inertia(ordinal_X, centers)

    np.testing.assert_array_equal(labels, np.array([0, 1, 2, 0]))
    np.testing.assert_allclose(inertia, 1/3.)


def test_initialize_frequencies(soybean_data):
    """test frequencies allign with orginal data."""
    X, labels = soybean_data
    n_samples, n_features = X.shape

    encoder = OrdinalEncoder(strategy='none')
    ordinal_X = encoder.fit_transform(X)
    max_n_levels = max(
            [len(levels) for levels in encoder.level_map])

    freq = k_modes._initialize_frequencies(
            ordinal_X, labels, 4, max_n_levels)

    assert freq.shape == (4, n_features, max_n_levels)

    # brute force check that counts are correct
    for label_id in range(4):
        label_frequencies = freq[label_id]
        for feature_id in xrange(n_features):
            uniques, counts = np.unique(ordinal_X[labels == label_id][:, feature_id],
                                        return_counts=True)
            feature_freq = label_frequencies[feature_id]
            nonzero_counts = feature_freq[feature_freq != 0]
            np.testing.assert_array_equal(counts, nonzero_counts)


def test_cluster_modes(soybean_data):
    """test modes calculation from frequency matrix is correct."""
    X, labels = soybean_data
    n_samples, n_features = X.shape

    encoder = OrdinalEncoder(strategy='none')
    ordinal_X = encoder.fit_transform(X)
    max_n_levels = max(
            [len(levels) for levels in encoder.level_map])

    freq = k_modes._initialize_frequencies(
            ordinal_X, labels, 4, max_n_levels)
    for label_id in range(4):
        modes = k_modes.cluster_modes(freq, label_id)
        expected_modes, expected_counts = stats.mode(
                ordinal_X[labels == label_id])
        np.testing.assert_array_equal(modes, expected_modes.ravel())


def test_count_matrix(soybean_data):
    X, labels = soybean_data
    n_samples, n_features = X.shape

    encoder = OrdinalEncoder(strategy='none')
    ordinal_X = encoder.fit_transform(X)
    max_n_levels = max(
            [len(levels) for levels in encoder.level_map])

    counts = k_modes.count_matrix(ordinal_X[0, :], max_n_levels)
    print(ordinal_X[0, :])
    print(counts)

def test_k_modes_soybean(soybean_data):
    #X = gen_data(n_samples=100000)
    X, labels = soybean_data

    clusterer = k_modes.KModes(n_clusters=4, init='cao', n_init=1,
                               random_state=123,
                               max_iter=10)
    #clusterer = kmodes.KModes(n_clusters=4, init='Cao', n_init=1, max_iter=100)
    t0 = time.time()
    clusterer.fit(X)
    print(time.time() - t0)
    ##print(clusterer.predict(X))
    #print(labels)
    print(clusterer.labels_)
    #print(clusterer.inertia_)
    #print(labels)
    #print('centers', clusterer.cluster_centers_)
    #print(clusterer.inertia_)

    #print np.mean(clusterer.labels_ == labels)
