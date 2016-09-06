import numpy as np
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


def anova_importance(X, cluster_labels, feature_names=None, n_features=5):
    """anova_importance

    Use the cluster ids as the dependent variables and do a one-way anova
    or t-test to determine significant deviations. May need to do (cluster
    not cluster) to do this on a per cluster basis.

    Returns
    -------
    importances: dict
        Returns a dict mapping cluster_id to list of top n features.
    """
    cluster_ids = np.unique(cluster_labels)
    for cluster_id in cluster_ids:
        # perform anova
        # select top n_features
        pass
