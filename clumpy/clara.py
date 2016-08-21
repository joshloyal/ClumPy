import numpy as np
from sklearn.utils import check_random_state

from clumpy import KMedoids


def expand_kmedoids(kmedoids, X):
    kmedoids.labels_ = kmedoids.predict(X)
    return kmedoids


def clara(X, n_iter=5, n_clusters=8, n_sub_samples=40, metric=None, random_state=1):
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
    best_inertia = np.inf
    best_kmedoids = None

    # we could run this in parallel?
    for i in range(n_iter):
        # subsequent samples should include the best mediods found in the previous steps.
        sample_idx = random_state.choice(np.arange(n_samples), size=sample_size, replace=False)
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=random_state)
        kmedoids.fit(X[sample_idx])

        # assign closest medoid to each datapoint
        inertia = kmedoids.inertia(X)

        if inertia < best_inertia:
            best_inertia = inertia
            best_kmedoids = kmedoids

    # now turn best_kmedoids into a kmedoids over the entire dataset
    kmedoids = expand_kmedoids(best_kmedoids, X)

    return kmedoids
