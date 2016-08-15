import sklearn

import clumpy
from clumpy import datasets
from clumpy import plots

import seaborn as sns

#X = datasets.fetch_hdbscan_demo()
X, y  = sklearn.datasets.make_blobs(n_samples=500, n_features=2, centers=2, random_state=123)
clusterer, v = clumpy.gmeans(X, n_jobs=-1, max_n_clusters=15)

#plots.plot_clusters(X, clusterer.labels_, clusterer.cluster_centers_)
