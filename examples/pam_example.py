import clumpy
from clumpy import datasets
from clumpy import plots

X = datasets.fetch_10kdiabetes_embedding()

clusterer = clumpy.KMedoids(random_state=1234).fit(X)
plots.plot_clusters(X, clusterer.labels_, clusterer.cluster_centers_)

