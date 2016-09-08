# *- coding: utf-8 -*-
"""K-Modes clustering"""

# Authors: Joshua Loyal <jloyal25@gmail.com>
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array


from clumpy.preprocessing import OrdinalEncoder


def _unique_rows(array):
    """http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
       should only be used for integer arrays (floating points can get screwed up
       with +/-0"""
    array = np.ascontiguousarray(array)
    flat_array = array.view(
            np.dtype((np.void, array.dtype.itemsize * array.shape[1])))
    return np.unique(flat_array).view(array.dtype).reshape(-1, array.shape[1])


def _haung_init(X, n_clusters, random_state=None):
    """Initialize centroids according to method of Huang [1997]."""
    random_state = check_random_state(random_state)

    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    for feature_idx in xrange(n_features):
        frequencies = np.bincount(X[:, feature_idx])
        labels = np.nonzero(frequencies)[0]

        # sample centers using probability of attributes
        centers[:, feature_idx] = random_state.choice(
                labels, n_clusters, p=(frequencies[labels]/n_samples))


    # set actual clusters to points in the dataset, i.e. find the closet
    # point in the dataset
    candidate_ids = []
    unique_X = _unique_rows(X)
    for k in xrange(n_clusters):
        distance_to_candidates = pairwise_distances(
                unique_X, centers[k, :], metric='hamming')
        candidate_idxs = np.argpartition(
                distance_to_candidates.ravel(), n_clusters-1)[:n_clusters]

        for candidate in candidate_idxs:
            if candidate not in candidate_ids:
                candidate_ids.append(candidate)
                centers[k, :] = unique_X[candidate, :]
                break

    return centers


def _init_centroids(X, n_clusters=8, init='cao', random_state=None):
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if isinstance(init, string_types) and init == 'cao':
        centers = _cao_init
    elif isinstance(int, string_types) and init == 'huang':
        centers = _huang_init
    else:
        raise ValueError("The init parameter for k-modes should"
                         " be 'cao' or 'haung' or an ndarray,"
                         " '%s' (type '%s') was passed." % (init, type(init)))


    return centers


def k_modes_single(X, n_clusters=8, init='cao',
                   max_iter=100, verbose=False,
                   tol=1e-4, random_state=None):
    random_state = check_random_state(random_state)

    best_labels, best_inertia, best_centers = None, None, None

    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state)

    if verbose:
        print("Initialization complete")




def k_modes(X, n_clusters=8, init='cao',
            n_init=1, max_iter=100, verbose=False,
            tol=1e-4, random_state=None, n_jobs=1,
            return_n_iter=False):
    """K-modes clustering algorithm"""
    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)

    if max_iter <= 0:
        raise ValueError("Number of iterations should be a positive number,"
                         " got %d instead" % max_iter)

    best_inertia = np.inf
    # tol = _tolerance(X, tol)  # data dependent tolerance

    # Are there more n_clusters than unique rows? Then set the unique
    # rows to the initial values and skip iteration.
    unique_X = _unique_rows(X)
    if unique_X.shape[0] <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = n_unique
        init = unique

    if hasattr(init, '__array__'):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)
        if n_init != 1:
            warnings.warn(
                    'Explicit initial center position passed: '
                    'performing only one init in k-modes instead of n_init=%d'
                    % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    best_labels, best_inertia, best_centers = None, None, None
    if n_jobs == 1:
        for it in range(n_init):
            labels, inertia, centers, n_iter_ = kmeans_single()
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


class KModes(BaseEstimator, ClusterMixin, TransformerMixin):
    """KModes.
    """
    def __init__(self, n_clusters=8, init='cao', n_init=1,
                 max_iter=100, tol=1e-4, verbose=0,
                 random_state=None, n_jobs=1):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs


    def _check_fit_data(self, X):
        X = check_array(X, dtype=None)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))


    def fit(self, X, y=None):
        """Compute k-modes clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        self.encoder_ = OrdinalEncoder(strategy='frequency')
        ordinal_X = self.encoder_.fit_transform(X)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
                k_modes(
                    ordinal_X, n_clusters=self.n_clusters, init=self.init,
                    n_init=self.n_init, max_iter=self.max_iter,
                    verbose=self.verbose,
                    tol=self.tol, random_state=self.random_state,
                    n_jobs=self.n_jobs, return_n_iter=True)

        return self
