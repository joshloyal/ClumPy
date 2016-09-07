from __future__ import division

import collections

import numpy as np
import scipy.stats as stats

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


def top_k_features(estimator, features=None, top_k=None):
    """top_k features from a forest ensemble."""
    importances = estimator.feature_importances_
    sorted_features = np.argsort(importances)[::-1]
    if top_k is not None:
        sorted_features = sorted_features[:top_k]

    if features is not None:
        return [features[index] for index in sorted_features]
    return sorted_features


def ova_forest_importance(X, cluster_labels, features=None, top_k=None):
    """Determine distinguishing cluster features based on
    RandomForest feature importances.
    """
    # fit a One-Vs-Rest classifier to distinguish clusters
    cluster_classifier = OneVsRestClassifier(
        estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1))
    cluster_classifier.fit(X, cluster_labels)

    feature_importance = [top_k_features(estimator,
                                         features=features,
                                         top_k=top_k) for estimator in
                          cluster_classifier.estimators_]

    return feature_importance


def ttest_importance(X,
                     cluster_labels,
                     feature_names=None,
                     n_features=5):
    """ttest_importance

    t-test takes clusters vs. overall statistics and looks at
    deviations of each variable between there means.

    Returns
    -------
    importances: dict
        Returns a dict mapping cluster_id to list of top n features.
    """
    importances = {}
    n_features = X.shape[1]
    cluster_ids = np.unique(cluster_labels)
    for cluster_id in cluster_ids:
        cluster_mask = (cluster_labels == cluster_id)
        in_cluster = X[cluster_mask]
        out_cluster = X#[~cluster_mask]
        pvalues = np.empty(n_features)
        for col in xrange(n_features):
            pvalues[col] = stats.ttest_ind(
                    in_cluster[:, col], out_cluster[:, col])[1]

        # apply beferoni correction and return lowest p-values (valid?)
        sig_mask = pvalues < (0.05 / n_features)
        top_k = np.argsort(pvalues[sig_mask])[:n_features]
        importances[cluster_id] = [
                feature for idx, feature in enumerate(feature_names) if
                idx in top_k]

    return importances


def anova_importance(X, cluster_labels, feature_names=None, n_features=5):
    """anova_importance

    Use the cluster ids as the dependent variables and do a one-way anova
    or t-test to determine significant deviations. May need to do (cluster
    not cluster) to do this on a per cluster basis.

    ANOVA takes variable vs. cluster id and determines significance.

    Returns
    -------
    importances: dict
        Returns a dict mapping cluster_id to list of top n features.
    """
    importances = {}
    cluster_ids = np.unique(cluster_labels)
    for cluster_id in cluster_ids:
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X, cluster_labels == cluster_id)

        if feature_names:
            importances[cluster_id] = [feature_names[support_id] for
                                       support_id in
                                       selector.get_support(indices=True)]
        else:
            importances[cluster_id] = selector.get_support(indices=True)
    return importances

def relevance_score(cluster_proba, marginal_proba, alpha):
    if cluster_proba == 0.0 and marginal_proba == 0.0:
        return np.nan
    else:
        return (alpha * np.log(cluster_proba) +
                (1 - alpha) * np.log(cluster_proba / marginal_proba))



def single_cluster_relevance(
        column, cluster_labels, cluster_id, data,
        marginal_probas=None, n_features=5, alpha=0.3):
    X = data[column].values
    levels = np.unique(X)

    if marginal_probas is None:
        n_samples = X.shape[0]
        levels = np.unique(X)
        marginal_probas = {}
        for level in levels:
            marginal_probas[level] = X[X == level].shape[0] / n_samples

    cluster_X = X[cluster_labels == cluster_id]
    n_samples_cluster = cluster_X.shape[0]

    rel = {}
    for level in levels:
        cluster_proba = cluster_X[cluster_X == level].shape[0] / n_samples_cluster
        rel[level] = relevance_score(
                cluster_proba, marginal_probas[level], alpha)

    rel = sorted(rel.items(), key=lambda x: x[1])[::-1]
    return [r[0] for r in rel if np.isfinite(r[1])][:n_features]


def categorical_relevance(column, cluster_labels, data, n_features=5, alpha=0.3):
    X = data[column].values
    cluster_ids = np.unique(cluster_labels)

    # calculate marginal statistics
    n_samples = X.shape[0]
    levels = np.unique(X)
    marginal_probas = {}
    for level in levels:
        marginal_probas[level] = X[X == level].shape[0] / n_samples


    relevance = {cluster_id: single_cluster_relevance(
                    column, cluster_labels, cluster_id, data,
                    marginal_probas=marginal_probas) for
                 cluster_id in cluster_ids}

    return relevance
