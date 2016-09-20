from __future__ import division

from warnings import warn

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_array, safe_mask
from sklearn.utils.validation import check_is_fitted


class UniqueThreshold(BaseEstimator, SelectorMixin):
    def __init__(self, max_frequency=1.0):
        self.max_frequency = max_frequency

    def fit(self, X, y=None):
        n_samples, n_features = X.shape

        self.unique_frequency_ = np.empty(n_features, dtype=np.float64)
        for column_id in xrange(n_features):
            self.unique_frequency_[column_id] = np.unique(X[:, column_id]).shape[0] / n_samples

        return self

    def transform(self, X):
        mask = self.get_support()
        if not mask.any():
            warn('No features were selected: either the data is'
                 ' too noisy or the slection test is too strict.',
                 UserWarning)
            return np.empty(0).reshape((X.shape[0], 0))
        if len(mask) != X.shape[1]:
            raise ValueError('X has a different shape than during fitting.')

        return X[:, safe_mask(X, mask)]


    def _get_support_mask(self):
        check_is_fitted(self, 'unique_frequency_')

        return self.unique_frequency_ < self.max_frequency

    def get_support(self, indices=False):
        mask = self._get_support_mask()
        return mask if not indices else np.where(mask)[0]



if __name__ == '__main__':
    rng = np.random.RandomState(10)
    X = np.hstack((np.arange(10).reshape(-1, 1),
                   rng.choice([0, 1], 10).reshape(-1, 1),
                   rng.randn(10).reshape(-1, 1)))

    print(UniqueThreshold(max_frequency=0.99).fit_transform(X))

