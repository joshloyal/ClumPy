# *- coding: utf-8 -*-
"""K-Modes clustering"""

# Authors: Joshua Loyal <jloyal25@gmail.com>
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from six import string_types
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


def _huang_init(X, n_clusters, random_state=None):
    """Initialize centroids according to method of Huang [1997]."""
    random_state = check_random_state(random_state)

    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    for feature_id in xrange(n_features):
        uniques, counts = np.unique(X[:, feature_id], return_counts=True)

        # sample centers using probability of attributes
        centers[:, feature_id] = random_state.choice(
                uniques, n_clusters, p=(counts/n_samples))


    # set actual clusters to points in the dataset, i.e. find the closet
    # point in the dataset
    candidate_ids = []
    unique_X = _unique_rows(X)
    for k in xrange(n_clusters):
        distance_to_candidates = pairwise_distances(
                unique_X, np.atleast_2d(centers[k, :]), metric='hamming')
        candidate_idxs = np.argpartition(
                distance_to_candidates.ravel(), n_clusters-1)[:n_clusters]

        for candidate in candidate_idxs:
            if candidate not in candidate_ids:
                candidate_ids.append(candidate)
                centers[k, :] = unique_X[candidate, :]
                break

    return centers


def _categorical_density(X):
    n_samples, n_features = X.shape

    # calculate density matrix for each point
    density = np.zeros(n_samples)
    for feature_id in xrange(n_features):
        uniques, counts = np.unique(X[:, feature_id], return_counts=True)
        density += counts[X[:, feature_id]]

    return density


def _cao_init(X, n_clusters, random_state=None):
    """Initialize centroids according to the method of Cao et al. [2009]

    Note: complexity is O(n_samples * n_features * n_clusters**2), i.e.
          linear in the number of samples and features and quadratic
          wrt the number of clusters.
    """
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=np.float32)

    # calculate density matrix for each point
    density = _categorical_density(X)

    # choose initial centroid based on distance and density
    centers[0, :] = X[np.argmax(density), :]

    if n_clusters > 1:
        for center_id in xrange(1, n_clusters):
            center_density = (
                    pairwise_distances(X, centers[:center_id, :]) *
                    density.reshape(-1, 1))
            centers[center_id, :] = X[np.argmax(np.min(center_density, axis=1))]

    return centers


def _init_centroids(X, n_clusters=8, init='cao', random_state=None):
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if isinstance(init, string_types) and init == 'cao':
        centers = _cao_init(X, n_clusters, random_state)
    elif isinstance(init, string_types) and init == 'huang':
        centers = _huang_init(X, n_clusters, random_state)
    else:
        raise ValueError("The init parameter for k-modes should"
                         " be 'cao' or 'huang' or an ndarray,"
                         " '%s' (type '%s') was passed." % (init, type(init)))


    return centers


def k_modes_single(X, n_clusters=8, init='cao',
                   max_iter=100, max_n_levels=None,
                   verbose=False,
                   tol=1e-4, random_state=None):
    random_state = check_random_state(random_state)

    best_labels, best_inertia, best_centers = None, None, None

    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state)

    if verbose:
        print("Initialization complete")


    # an array of frequencies of each attribute in each cluster
    frequencies = None
    old_labels = None

    for i in range(max_iter):
        # assign cluster labels to points (E-step of EM)
        labels, inertia = _labels_inertia(X, centers)
        print(labels)

        # computation of the modes (M-step of EM)
        centers, frequencies = _k_modes_centers(
            X, labels, old_labels, n_clusters, frequencies, max_n_levels, random_state)

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        if old_labels is not None and (old_labels != labels).sum() == 0:
            break

        old_labels = labels

    return best_labels, best_inertia, best_centers, i + 1



def _initialize_frequencies(X, labels, n_clusters, max_n_levels):
    """Initialize a frequency array. This is an array of shape
    (n_clusters, n_features, max_n_levels) where
    frequency[cluster_id] gives you the frequency table of that
    cluster.
    """
    n_features = X.shape[1]
    frequencies = np.zeros((n_clusters, n_features, max_n_levels))

    # what is faster? (probably good for large features?)
    for cluster_id in xrange(n_clusters):
        cluster_mask = (labels == cluster_id)
        counts = np.eye(max_n_levels)[X[cluster_mask]].sum(axis=0)
        frequencies[cluster_id] += counts

    #for cluster_id in xrange(n_clusters):
    #    cluster_mask = (labels == cluster_id)
    #    for feature_id in xrange(n_features):
    #        uniques, counts = np.unique(
    #                X[cluster_mask][:, feature_id], return_counts=True)
    #        frequencies[cluster_id, feature_id, :len(uniques)] = counts

    return frequencies


def _move_points(X, frequencies, max_n_levels, old_labels, new_labels):
    """Move points between clusters. (this needs to be cythonized)"""
    moved_ids = np.where(old_labels != new_labels)
    for moved_id in moved_ids:
        moved_counts = np.eye(max_n_levels)[X[moved_id]]
        frequencies[old_labels[moved_id]] -= moved_counts
        frequencies[new_labels[moved_id]] += moved_counts


def _labels_inertia(X, centers):
    # cythonize me!
    n_samples = X.shape[0]
    n_clusters, n_features = centers.shape
    labels = np.empty(n_samples)
    inertia = 0

    # we need to actualy update modes after each allocation!
    for point_id in xrange(n_samples):
        distances = pairwise_distances(X[point_id], centers, metric='hamming')
        labels[point_id] = np.argmin(distances, axis=1)
        inertia += distances[labels[point_id]]
    #distances = pairwise_distances(X, centers, metric='hamming')
    #labels = np.argmin(distances, axis=1)

    #inertia = distances[np.arange(distances.shape[0]), labels].sum()

    return labels, inertia


def _k_modes_centers(X, labels, old_labels, n_clusters, frequencies, max_n_levels, random_state=None):
    random_state = check_random_state(random_state)
    n_features = X.shape[1]
    centers = np.empty((n_clusters, n_features))

    if frequencies is None:
        frequencies = _initialize_frequencies(
                X, labels, n_clusters, max_n_levels)
    else:
        _move_points(X, frequencies, max_n_levels, old_labels, labels)

    # centroid update
    for center_id in xrange(n_clusters):
        counts = (labels == center_id).sum()
        if counts == 0:  # empty cluster random init from largest cluster
            #for feature_id in xrange(n_features):  # please refactor!
            #    uniques = np.unique(X[:, feature_id])
            #    centers[center_id, feature_id] = random_state.choice(uniques)
            # cache this maybe?
            cluster_ids, counts = np.uniques(labels, return_counts=True)
            max_cluster_id = counts.argmax()
            new_id = random_state.choice(labels[labels == max_cluster_id])
            centers[center_id, :] = X[new_id, :]

            moved_counts = np.eye(max_n_levels)[X[new_id]].sum(axis=0)
            frequency[max_cluster_id] -= moved_counts
            frequency[center_id] += moved_counts
            labels[new_id] = cluster_id
        else:  # init to new mode of cluster
            centers[center_id, :] = np.argmax(frequencies[center_id], axis=1)

    return centers, frequencies


def k_modes(X, n_clusters=8, init='cao',
            n_init=1, max_iter=100, verbose=False,
            tol=1e-4, random_state=None, n_jobs=1,
            return_n_iter=False, max_n_levels=None):
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

            labels, inertia, centers, n_iter_ = k_modes_single(
                X, n_clusters=n_clusters, init=init,
                max_iter=max_iter, max_n_levels=max_n_levels,
                verbose=verbose, tol=tol, random_state=random_state)

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
    def __init__(self, n_clusters=8, init='huang', n_init=1,
                 max_iter=100, tol=1e-4, verbose=False,
                 random_state=None, n_jobs=1):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose


    def _check_fit_data(self, X):
        X = check_array(X, dtype=None)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))

        return X

    def fit(self, X, y=None):
        """Compute k-modes clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        self.encoder_ = OrdinalEncoder(strategy='none')
        ordinal_X = self.encoder_.fit_transform(X)
        max_n_levels = max(
                [len(levels) for levels in self.encoder_.level_map])

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
                k_modes(
                    ordinal_X, n_clusters=self.n_clusters, init=self.init,
                    n_init=self.n_init, max_iter=self.max_iter,
                    max_n_levels=max_n_levels,
                    verbose=self.verbose,
                    tol=self.tol, random_state=self.random_state,
                    n_jobs=self.n_jobs, return_n_iter=True)

        return self
