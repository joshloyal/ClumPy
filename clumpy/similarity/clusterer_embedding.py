from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from sklearn.manifold import MDS


from clumpy.similarity import cluster_similarity


def to_dissimilarity_matrix(clusterers, X):
    """
    Parameters
    ----------
    clusters : list of sklearn style clusteres
    """
    n_clusterers = len(clusterers)
    dissimilarity_matrix = np.ones((n_clusterers, n_clusterers), dtype=np.float64)

    for alg_id_a, clusterer_a in enumerate(clusterers):
        for alg_id_b, clusterer_b in enumerate(clusterers):
            dissimilarity_matrix[alg_id_a, alg_id_b] = (1 - cluster_similarity(
                clusterer_a, clusterer_b, X))

    return dissimilarity_matrix


def clusterer_embedding(clusterers, X, random_state=123):
    dissimilarity_matrix = to_dissimilarity_matrix(clusterers, X)

    embedding = MDS(n_components=2,
                    dissimilarity='precomputed',
                    random_state=random_state).fit_transform(dissimilarity_matrix)

    return embedding
