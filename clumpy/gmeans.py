import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import randomized_svd
from sklearn.preprocessing import StandardScaler


def relative_improvement(reference_score, new_score):
    return float(reference_score - new_score) / reference_score


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


def make_new_clusterer(X, cluster_centers, **clusterer_kwargs):
    if 'n_clusters' in clusterer_kwargs:
        del clusterer_kwargs['n_clusters']
    n_clusters = cluster_centers.shape[0]
    clusterer = KMeans(n_clusters=n_clusters, **clusterer_kwargs)
    clusterer.cluster_centers_ = cluster_centers
    clusterer.labels_ = clusterer.predict(X)

    return clusterer


def split_center(parent_center_idx, child_centers, cluster_centers):
    cluster_centers = cluster_centers.tolist()
    cluster_centers.remove(cluster_centers[parent_center_idx])
    for center in child_centers:
        cluster_centers.append(center)

    return np.array(cluster_centers)


def split_clusterer(X, parent_center_idx, child_centers, clusterer):
    cluster_centers = split_center(parent_center_idx,
                                   child_centers,
                                   clusterer.cluster_centers_)
    return make_new_clusterer(X, cluster_centers, **clusterer.get_params())


def gmeans(X,
           clusterer=None,
           max_n_clusters=None,
           min_cluster_samples=100,
           min_improvement=0.1,
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
    if clusterer is None:  # initialize with data mean
        cluster_centers = np.mean(X, axis=0).reshape(-1, X.shape[1])
        clusterer = make_new_clusterer(
            X, cluster_centers, random_state=random_state, n_jobs=n_jobs)
    else:
        cluster_centers = clusterer.cluster_centers_

    cluster_labels = clusterer.labels_
    n_clusters = cluster_centers.shape[0]
    print('n_clusters: %i' % n_clusters)

    # Maximum number of clusters reached. Terminate splitting
    if max_n_clusters and n_clusters >= max_n_clusters:
        return clusterer

    # cache original cluster scores
    if np.unique(cluster_labels).shape[0] > 1:
        scores = metrics.silhouette_samples(X, cluster_labels, metric='euclidean')
        score = scores.mean()
    else:
        score = -1

    n_centers_added = 0
    new_k = 0
    for k, cluster_center in enumerate(cluster_centers):
        X_k = X[cluster_labels == k]
        if X_k.shape[0] <= min_cluster_samples:  # not enough points to split
            new_k = new_k + 1
            continue

        center_init = proposed_centers(X_k, cluster_center)
        kmeans = KMeans(
            n_clusters=2,
            init=center_init,
            random_state=random_state,
            n_init=1).fit(X_k)

        # project X onto the plane connecting the cluster centers
        v = kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]
        x_prime = np.dot(X_k, v) / np.linalg.norm(v, ord=2)
        x_prime = StandardScaler().fit_transform(x_prime.reshape(-1, 1))

        # anderson-darling test
        result = stats.anderson(x_prime.ravel(), dist='norm')
        statistic = result.statistic
        critical_value = result.critical_values[-1]

        if np.abs(statistic) > np.abs(critical_value):

            # this is wrong when we split the cluster k changes...
            clusterer_candidate = split_clusterer(
                X,
                parent_center_idx=new_k,
                child_centers=kmeans.cluster_centers_,
                clusterer=clusterer)

            new_score = metrics.silhouette_score(X,
                                                 clusterer_candidate.labels_,
                                                 metric='euclidean')
            improvement = relative_improvement(score, new_score)
            if improvement > min_improvement:
                print('splitting: statistic=%.2f, critical_value=%.2f' % (statistic, critical_value))
                print("increase k (%2.2f%%)" % (improvement * 100))
                clusterer = clusterer_candidate
                n_centers_added += 1
                continue
        new_k = new_k + 1

    if n_centers_added:
        return gmeans(X,
                      clusterer,
                      max_n_clusters=max_n_clusters,
                      random_state=random_state,
                      n_jobs=n_jobs)

    return clusterer
