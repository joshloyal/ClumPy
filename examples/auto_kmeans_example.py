import clumpy
from clumpy import datasets
from clumpy import plots

X = datasets.fetch_hdbscan_demo()
clusterer = clumpy.auto_kmeans(X, n_clusters=[2, 3, 24], n_jobs=-1)

plots.plot_clusters(X, clusterer.labels_)
