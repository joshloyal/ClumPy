"""Python implementation of the R's daisy function
"""
import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances


random_state = check_random_state(0)
numeric_X = random_state.randn(100, 10)


def min_max_scale(X):
    data_min = np.min(X, axis=0)
    data_max = np.max(X, axis=0)
    return X / (data_max - data_min)


def gower_numeric(X, n_jobs=1):
    scaled_X = min_max_scale(X)
    return pairwise_distances(scaled_X, metric='manhattan', n_jobs=n_jobs)


def gower_categorical(X, n_jobs=1):
    return pairwise_distances(X, metric='hamming', n_jobs=n_jobs)


class GowerDistance(object):
    def __init__(self,
                 numeric_indices=None,
                 categorical_indices=None,
                 gamma='heuristic',
                 n_jobs=1):
        self.numeric_indices = numeric_indices
        self.categorical_indices = categorical_indices
        self.gamma = gamma
        self.n_jobs = n_jobs
        self._n_features = None

    @property
    def has_numerics(self):
        return self.numeric_indices is not None

    @property
    def has_categoricals(self):
        return self.categorical_indices is not None

    @property
    def n_features_(self):
        if self._n_features is not None:
            return self._n_features

        self._n_features = 0
        if self.has_numerics:
            self._n_features += len(self.numeric_indices)
        if self.has_categoricals:
            self._n_features += len(self.categorical_indices)

        return self._n_features

    def __call__(self, X):
        n_samples, n_features = X.shape
        assert n_features == self.n_features_

        diss = np.zeros((n_samples, n_samples))

        if self.gamma == 'heuristic' and self.has_numerics:
            self.gamma = 0.5 * X[:, self.numeric_indices].std()


        if self.has_numerics:
            diss += gower_numeric(X[:, self.numeric_indices],
                                  n_jobs=self.n_jobs)

        if self.has_categoricals:
            gamma = self.gamma if self.has_numerics else 1.0
            diss += gamma * gower_categorical(X[:, self.categorical_indices],
                                              n_jobs=self.n_jobs)

        return diss
