import numpy as np
from sklearn.decomposition import RandomizedPCA, MiniBatchSparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, MinMaxScaler

import clumpy


class Cluster(object):
    def __init__(self, numeric_columns=[], categorical_columns=[]):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.clusterer_ = None
        self.importances_ = None

    #@property
    #def feature_names(self):
    #    return self.numeric_columns + self.categorical_columns

    @property
    def n_clusters(self):
        return self.clusterer_.n_clusters

    def find_clusters(self, df):
        X = np.hstack([X for X in clumpy.preprocessing.process_data(df) if X is not None])

        # reduction using pca
        #pca = RandomizedPCA(n_components=50, random_state=123, iterated_power=7)
        pca = TruncatedSVD(n_components=50, random_state=123)
        scaled_X = pca.fit_transform(X)
        scaled_X = MinMaxScaler().fit_transform(scaled_X)
        #pca = MiniBatchSparsePCA(n_components=50, alpha=0.8, n_iter=100, random_state=123)
        #scaled_X = np.hstack((X[:, :len(num_columns)], pca_X))
        #scaled_X = scaled_X - np.mean(scaled_X, axis=0)
        #max_x = np.max(np.abs(scaled_X), axis=0)
        #max_x[max_x == 0] = 1.
        #scaled_X = scaled_X / max_x
        #ptp_scale = np.ptp(scaled_X, axis=0)
        #ptp_scale[ptp_scale == 0] = 1.
        #scaled_X /= ptp_scale
        #scaled_X = normalize(scaled_X, norm='l2', axis=1, copy=False)

        #self.clusterer_ = clumpy.cluster.auto_kmeans(scaled_X, n_clusters=[2, 3, 4])
        #self.find_rules(X)
        #self.rules_ = clumpy.rules.prim_descriptions(
        #        data[self.numeric_columns + self.categorical_columns], self.clusterer_.labels_, feature_names=self.importances_)
        ##self.rules_ = clumpy.rules.tree_descriptions(
        #        data[self.feature_names], self.clusterer_.labels_,
        #        categorical_columns=self.categorical_columns,
        #        feature_names=self.importances_)
        tsne = TSNE(n_components=2, random_state=1234, verbose=True)
        self.embedding_ = tsne.fit_transform(scaled_X)
        self.embedding_ -= np.mean(self.embedding_, axis=0)

        #self.clusterer_ = clumpy.cluster.auto_kmeans(self.embedding_, n_clusters=[2, 3, 4])


    def find_rules(self, X):
        self.importances_ = clumpy.importance.anova_importance(
                X,
                self.clusterer_.labels_,
                feature_names=self.feature_names,
                n_features=5)



def cluster(X, numeric_columns=None, categorical_columns=None):
    clusterer = Cluster(numeric_columns=numeric_columns,
                        categorical_columns=categorical_columns)

    clusterer.find_clusters(X)


    return clusterer


def plot(clusterer, data, cluster_id):
    cluster_importances = clusterer.importances_[cluster_id]
    cat_vars = [var for var in cluster_importances if var in clusterer.categorical_columns]
    num_vars = [var for var in cluster_importances if var in clusterer.numeric_columns]

    return clumpy.plots.plot_cluster_statistics(
            cluster_labels=clusterer.clusterer_.labels_,
            cluster_id=cluster_id,
            data=data,
            scale=True,
            quant_var=num_vars,
            qual_var=cat_vars,
            figsize=(15,15))

