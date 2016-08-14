import clumpy
from clumpy import datasets
from clumpy import plots

X = datasets.fetch_hdbscan_demo()
clusterer = clumpy.incremental_kmeans(X, n_jobs=-1, max_n_clusters=15)

plots.plot_clusters(X, clusterer.labels_)
