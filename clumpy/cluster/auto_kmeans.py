from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import numpy as np
from sklearn import datasets, cluster, metrics
from sklearn.model_selection import GridSearchCV


class UnsupervisedCV(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def split(self, X=None, y=None, groups=None):
        return np.arange(self.n_samples), np.arange(self.n_samples)

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def __len__(self):
        return self.get_n_splits()

    def __iter__(self):
        yield self.split()


def make_clusterer_truth_scorer(metric):
    def clusterer_scorer(estimator, X, y):
        return metric(y, estimator.labels_)

    return cluster_scorer


class _ClusterScorer(object):
    def __init__(self, score_func):
        self._score_func = score_func
        self.sign = 1

    def __call__(self, estimator, X):
        cluster_labels = estimator.labels_
        return self.sign * self._score_func(X, cluster_labels)


def make_cluster_coherence_scorer(metric):
    return _ClusterScorer(metric)


def auto_kmeans(X, n_clusters=[2, 3, 4], n_jobs=1):
    """auto_keans.

    Fit a KMeans model with various values of `K` and choose the best
    value of K based on the best silhoette score. This could be done
    in parallel instead of sequential; however, we take advantage of
    the parallelism inside the model instead.
    """
    best_score = -np.inf
    best_clusterer = None
    for cluster_k in n_clusters:
        cluster_estimator = cluster.KMeans(n_clusters=cluster_k, n_init=5, max_iter=10, n_jobs=n_jobs)
        labels = cluster_estimator.fit_predict(X)
        score = metrics.silhouette_score(X, labels)
        if score > best_score:
            best_clusterer = cluster_estimator
            best_score = score

    return best_clusterer


