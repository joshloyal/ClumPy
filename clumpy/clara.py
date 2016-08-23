import numpy as np

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.utils import check_random_state

from clumpy import KMedoids


def expand_kmedoids(kmedoids, X):
    kmedoids.labels_ = kmedoids.predict(X)
    return kmedoids


def _fit_kmedoids(kmedoids, X, sample_ids):
    kmedoids_ = clone(kmedoids)
    kmedoids_.fit(X[sample_ids])
    inertia = kmedoids_.inertia(X)

    return (inertia, kmedoids_)


def clara_sample_generator(n_iter, X, sample_size, kmedoids, random_state):
    n_samples = X.shape[0]
    for i in xrange(n_iter):
        sample_ids = random_state.choice(
                np.arange(n_samples), size=sample_size, replace=False)
        yield (kmedoids, X, sample_ids)


def clara(X, n_iter=5, n_clusters=8, n_sub_samples=40, metric=None, n_jobs=1, random_state=1):
    """CLARA (Clustering Large Applications).

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)

    n_iter : int
        Number of iterations of CLARA

    n_clusters : int, optional, default: 8
        How many medoids. Must be positive.

    n_sub_samples : int
        Number of samples used in each CLARA iteration to build the medoids

    metric : string, optional, default: 'euclidean'
        What distance metric to use.

    random_state : int, optional, default: None
        Specify random state for the random number generator.

    Returns
    -------
    kmediods : KMedoids
        Best kmedoids object constructed by CLARA
    """

    random_state = check_random_state(random_state)
    n_samples = X.shape[0]
    sample_size = min(n_samples, n_sub_samples + 2 * n_clusters)
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=random_state)

    # fit subsampled k-medoids in parallel.
    # NOTE: Doesn't work for original heuristic that uses medoids found in
    #       the previous step, but could be re-initialized for each
    #       new parallel batch.
    result = Parallel(n_jobs=n_jobs)(delayed(_fit_kmedoids)(*x) for x in
            clara_sample_generator(n_iter, X, sample_size, kmedoids, random_state))

    # select kmedoids with the lowest inertia
    best_kmedoids = sorted(result)[0][1]

    # now turn best_kmedoids into a kmedoids over the entire dataset
    kmedoids = expand_kmedoids(best_kmedoids, X)

    return kmedoids
