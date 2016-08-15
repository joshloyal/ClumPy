import clumpy
from clumpy import datasets
from clumpy import plots

X = datasets.fetch_10kdiabetes_embedding()

clusterer = clumpy.auto_kmeans(X,
                               n_clusters=[3],
                               n_jobs=1)
plots.plot_clusters(X, clusterer.labels_, clusterer.cluster_centers_)
