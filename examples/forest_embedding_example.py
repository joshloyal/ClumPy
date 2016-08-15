from sklearn import manifold
from sklearn import decomposition
from sklearn.cluster import KMeans

import clumpy
from clumpy import datasets
from clumpy import plots

X = datasets.fetch_hdbscan_demo()

print('tree embedding')
n_trees = 5000
rf = clumpy.RandomForestEmbedding(n_estimators=n_trees, random_state=10, n_jobs=-1, sparse_output=True)
leaves = rf.fit_transform(X)


print('projection')
if leaves.shape[1] > 50:
    projection = decomposition.TruncatedSVD(n_components=50, random_state=123).fit_transform(leaves)
else:
    projection = leaves.toarray()
projector = manifold.TSNE(random_state=1234, init='pca')
embedding = projector.fit_transform(projection)


print('clustering')
clusterer = KMeans(n_clusters=4, random_state=1234, n_init=5)
clusterer.fit(embedding)

plots.plot_clusters(X, clusterer.labels_)
