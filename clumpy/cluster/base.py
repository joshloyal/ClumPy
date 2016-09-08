import numpy as np
from sklearn.cluster import KMeans


def convert_to_kmeans(X, clusterer, params=None):
    cluster_centers = []
    for label in np.unique(clusterer.labels_):
        if label == -1:
            continue
        X[clusterer.labels_ == label]
        cluster_centers.append(X[clusterer.labels_ == label].mean(axis=0))
    return init_kmeans(X, np.array(cluster_centers), params=params)


def init_kmeans(X, cluster_centers, params=None):
    params = {} if params is None else params

    if 'n_clusters' in params:
        del params['n_clusters']

    n_clusters = cluster_centers.shape[0]
    clusterer = KMeans(n_clusters=n_clusters, **params)
    clusterer.cluster_centers_ = cluster_centers
    clusterer.labels_ = clusterer.predict(X)

    return clusterer
