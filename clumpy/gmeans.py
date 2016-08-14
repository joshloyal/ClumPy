import numpy as np
import scipy.stats as stats
from sklearn import cluster
from sklearn.decomposition import randomized_svd


def randomized_pca(X, n_components,
                   n_iter=3,
                   random_state=1234):

    mean_ = np.mean(X, axis=0)
    X -= np.mean(X, axis=0)
    U, S, V = randomized_svd(X, n_components,
                             n_iter=n_iter,
                             random_state=random_state)

    components = V
    eigen_values = np.diag(S)

    return components, eigen_values


def gmeans(X,
           clusterer=None,
           max_n_clusters=30,
           random_state=1234,
           n_jobs=1):
    """gmeans.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        The dataset to cluster.

    clusterer : scikit-learn style clusterer or None
        A pre-fitted cluster class (KMeans) fitted on the dataset `X`.
        If None then fit a KMeans cluster based on `auto_kmeans`.

    max_n_clusters : int
        Maximum number of clusters to form before terminating the iteration.
        Set to None to continue indefintiely until `min_improvement` is met.

    random_state : int
        Seed to the random number generator

    n_jobs : int
        Number of jobs to run in parallel. Used by the clustering object.
    """
    pca = randomized_pca(X, n_components=1, random_state=random_state)
    return pca
