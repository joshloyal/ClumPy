import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import randomized_svd
from sklearn.preprocessing import StandardScaler



def first_principal_component(X, n_iter=3, random_state=1234):
    mean_ = np.mean(X, axis=0)
    X -= np.mean(X, axis=0)
    _, S, V = randomized_svd(X, n_components=1,
                             n_iter=n_iter,
                             random_state=random_state)
    return V[0], S[0]


def proposed_centers(X, center):
    principal_component, eigen_value = first_principal_component(X)
    m = principal_component * np.sqrt(2 * eigen_value / np.pi)
    c1 = center + m
    c2 = center - m

    return np.vstack((c1, c2))


def split_center(old_center, new_centers, clusterer):
    n_clusteres = clusterer.cluster_centers_.shape[0]
    new_kmeans = KMeans(n_clusters=n_clusters+1, random_state=1234)
    new_centers = clusterer.cluster_centers_.tolist()
    new_centers.remove(new_centers[])

    return new_kmeans

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
    if clusterer is None:
        cluster_centers = np.mean(X, axis=0).reshape(-1, X.shape[1])
        cluster_labels = np.zeros(X.shape[0])
        n_clusters = cluster_centers.shape[0]
    else:
        cluster_centers = clusterer.cluster_centers_
        cluster_labels = clusterer.labels_
        n_clusters = cluster_centers.shape[0]

    n_centers_added = 0
    for k, cluster_center in enumerate(cluster_centers):
        X_k = X[cluster_labels == k]
        if len(X_k) <= 2:  # not enough datapoints to split (hyper-parameter?)
            continue

        center_init = proposed_centers(X_k, cluster_center)
        kmeans = KMeans(n_clusters=2, init=center_init, random_state=random_state, n_init=1)
        kmeans.fit(X_k)

        v = kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]
        x_prime = np.dot(X_k, v) / np.linalg.norm(v, ord=2)
        x_prime = StandardScaler().fit_transform(x_prime.reshape(-1, 1))

        result = stats.anderson(x_prime.ravel(), dist='norm')
        statistic = result.statistic
        critical_value = result.critical_values[-1]

        if statistic > critical_value:
            new_kmeans = add_centers(cluster, kmeans.cluster_centers_)
            n_centers_added += 1

    if n_centers_added:
        return gmeans(clusterer)

    return clusterer
