import clumpy
from clumpy import datasets
from clumpy import plots

#X = datasets.fetch_hdbscan_demo()
X = datasets.fetch_10kdiabetes_embedding()

clusterer = clumpy.gmeans(X, max_n_clusters=5, n_jobs=1)

plots.plot_clusters(X, clusterer.labels_, clusterer.cluster_centers_)
