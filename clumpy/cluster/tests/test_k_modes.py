import time

import numpy as np
import pandas as pd
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


def test_k_modes_centers():
    pass


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
    #print(clusterer.labels_)
    #print(clusterer.inertia_)
    #print(labels)
    #print('centers', clusterer.cluster_centers_)
    #print(clusterer.inertia_)


    @np.vectorize
    def to_labels(X):
        label_dict = {'D1': 1, 'D2': 2, 'D3': 0, 'D4': 3}
        return label_dict[X]

    print np.mean(clusterer.labels_ == to_labels(labels))

