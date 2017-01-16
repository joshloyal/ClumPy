from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import networkx as nx
import numpy as np

from clumpy.similarity import jaccard_similarity



def get_induced_partitions(clusterer, data):
    """Return the partition of the dataset induced by a clustering algorithm.

    Parameters
    ----------
    clusterer : sklearn style clustering algorithms
        This clusterer will be used to partition in the input data.

    data : array-like of shape [n_samples, n_features]
        The data that the clusterer will label.

    Returns:
    --------
    A list of length clusterer.n_clusters. Each element is the indices
    of the data points placed in that cluster.
    """
    if hasattr(clusterer, 'predict'):
        labels = clusterer.predict(data)
    else:
        labels = clusterer.labels_
        if labels.shape[0] != data.shape[0]:
            raise ValueError('Could not get predictions')

    return [np.where(labels == cluster_id)[0]
            for cluster_id in xrange(clusterer.n_clusters)]


def to_similarity_matrix(clusterer_a, clusterer_b, data):
    partitions_a = get_induced_partitions(clusterer_a, data)
    partitions_b = get_induced_partitions(clusterer_b, data)

    n_clusters_a = clusterer_a.n_clusters
    n_clusters_b = clusterer_b.n_clusters
    S = np.zeros((n_clusters_a, n_clusters_b), dtype=np.float64)
    for cluster_id_a, part_a in enumerate(partitions_a):
        for cluster_id_b, part_b in enumerate(partitions_b):
            S[cluster_id_a, cluster_id_b] = jaccard_similarity(part_a, part_b)

    return S


def to_adjacency_matrix(similarity_matrix):
    n_vertices_U, n_vertices_V = similarity_matrix.shape
    n_vertices = n_vertices_U + n_vertices_V
    adjacency_matrix = np.zeros((n_vertices, n_vertices), dtype=np.float64)

    # fill the adjacency matrix
    adjacency_matrix[:n_vertices_U, n_vertices_U:] = similarity_matrix
    adjacency_matrix[n_vertices_U:, :n_vertices_U] = similarity_matrix.T

    return adjacency_matrix


def cluster_similarity(clusterer_a, clusterer_b, data):
    similarity_matrix = to_similarity_matrix(clusterer_a, clusterer_b, data)
    graph = nx.from_numpy_matrix(to_adjacency_matrix(similarity_matrix))

    max_matching = nx.max_weight_matching(graph)

    return np.mean([graph.edge[vertex_id][max_matching[vertex_id]]['weight']
                    for vertex_id in graph if vertex_id in max_matching])
