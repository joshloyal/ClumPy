import numpy as np
import clumpy
from clumpy import datasets
from clumpy import plots
from clumpy.gmeans import make_new_clusterer

import hdbscan
X = datasets.fetch_10kdiabetes_embedding()


cluster_centers = []
clusterer = hdbscan.HDBSCAN(min_cluster_size=100).fit(X)
for label in np.unique(clusterer.labels_):
    if label == -1:
        continue
    X[clusterer.labels_ == label]
    cluster_centers.append(X[clusterer.labels_ == label].mean(axis=0))
clusterer = make_new_clusterer(X, np.array(cluster_centers))
#plots.plot_clusters(X, clusterer.labels_, clusterer.cluster_centers_)

print('fitting rules')
importances = clumpy.rules.ova_forest_importance(X, clusterer.labels_, top_k=1)
