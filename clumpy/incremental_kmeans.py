import numpy as np
from sklearn import metrics
from sklearn import cluster

from clumpy.auto_kmeans import auto_kmeans


def compute_improvement(ref, new):
    return float(ref - new) / ref


def incremental_kmeans(X,
                       clusterer=None,
                       min_improvement=0.01,
                       min_cluster_samples=None,
                       max_n_clusters=30,
                       random_state=1234,
                       n_jobs=1):
    """incremental_kmeans.

    Split worst cluster based on the silhouette score. This algorithm
    proceeds as follows:

        1. Start with n_clusters = K
        2. Identify the worst cluster based on an unsupervised measure
           (e.g. silhouette score)
        3. Split the worst cluster into 2 sub-clusters
        4. Measure the global improvement with the new clusters
        5. If you get an improvement continue adding clusters.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        The dataset to cluster.

    clusterer : scikit-learn style clusterer or None
        A pre-fitted cluster class (KMeans) fitted on the dataset `X`.
        If None then fit a KMeans cluster based on `auto_kmeans`.

    min_improvement : float
        Minimum score improvement to continue incrementing the clusters

    max_n_clusters : int
        Maximum number of clusters to form before terminating the iteration.
        Set to None to continue indefintiely until `min_improvement` is met.

    random_state : int
        Seed to the random number generator

    n_jobs : int
        Number of jobs to run in parallel. Used by the clustering object.
    """
    if clusterer is None:
        clusterer = auto_kmeans(X, n_clusters=[2], n_jobs=n_jobs)

    n_clusters = clusterer.n_clusters
    if n_clusters >= max_n_clusters:
        return clusterer

    labels = clusterer.labels_

    # compute the distributions of the labels
    labels_ratio = np.histogram(labels, bins=np.unique(labels).shape[0])[0]
    labels_ratio = np.array(labels_ratio, dtype=np.float64) / labels.shape[0]
    scores = metrics.silhouette_samples(X, labels, metric='euclidean')
    score = scores.mean()

    # measure global performance of each cluster
    k_scores = np.zeros(n_clusters)
    for k in range(n_clusters):
        k_scores[k] = scores[np.where(labels == k)].mean()

    # identify the cluster to split
    index = np.where(labels_ratio > 0.01)[0]
    worst_score = k_scores[index].max()
    worst_index = np.where(k_scores == worst_score)[0]
    worst_k = worst_index[0]

    # split worst cluster
    X_k = X[np.where(labels == worst_k)[0]]
    if min_cluster_samples and len(X_k) <= min_cluster_samples:  # not enough datapoints to split
        return clusterer

    # split into two clusters
    kmeans = cluster.KMeans(
        n_clusters=2, random_state=2, n_jobs=n_jobs, n_init=5, max_iter=10)
    kmeans.fit(X_k)

    # measure improvement (if any)
    ikmeans = cluster.KMeans(n_clusters=n_clusters+1, random_state=random_state)
    new_centers = np.array(clusterer.cluster_centers_).tolist()
    new_centers.remove(new_centers[worst_k])
    for center in kmeans.cluster_centers_:
        new_centers.append(center)
    ikmeans.cluster_centers_ = np.array(new_centers)
    new_labels = ikmeans.predict(X)
    ikmeans.labels_ = new_labels
    new_score = metrics.silhouette_score(X, ikmeans.labels_, metric='euclidean')

    improvement = compute_improvement(score, new_score)

    if improvement > min_improvement:
        print("increase k (%2.2f%%)" % (improvement * 100))
        return incremental_kmeans(
            X, clusterer=ikmeans, min_improvement=min_improvement,
            max_n_clusters=max_n_clusters,
            random_state=random_state, n_jobs=n_jobs)
    else:
        print("best k = %i (score = %.3f)" % (n_clusters, new_score))
        return clusterer
