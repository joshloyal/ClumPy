import clumpy
from clumpy import datasets
from clumpy import plots

#X = datasets.fetch_hdbscan_demo()
X = datasets.fetch_10kdiabetes_embedding()
clusterer = clumpy.incremental_kmeans(X, n_jobs=1, max_n_clusters=15, min_cluster_samples=100)

plots.plot_clusters(X, clusterer.labels_, clusterer.cluster_centers_)
