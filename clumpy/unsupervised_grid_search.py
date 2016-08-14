import numpy as np
from sklearn import datasets, cluster, metrics
from sklearn.grid_search import GridSearchCV


class UnsupervisedCV(object):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def split(self):
        return np.arange(self.n_samples), np.arange(self.n_samples)

    def get_n_splits(self):
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



if __name__ == '__main__':
    X, _ = datasets.make_blobs(centers=3, n_samples=1000, random_state=1234)
    grid_search = GridSearchCV(
        cluster.KMeans(n_init=20, max_iter=10),
        param_grid={'n_clusters': [2, 3, 4]},
        cv=UnsupervisedCV(n_samples=int(X.shape[0])),
        scoring=make_cluster_coherence_scorer(metrics.silhouette_score),
        n_jobs=3)
    grid_search.fit(X)
    assert grid_search.best_params_['n_clusters'] == 3
