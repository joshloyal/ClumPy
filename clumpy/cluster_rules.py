import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


def get_code(tree, feature_names):
    """return code from a decision tree."""
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node):
        if (threshold[node] != -2):
            print "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
            if left[node] != -1:
                recurse (left, right, threshold, features,left[node])
            print "} else {"
            if right[node] != -1:
                recurse (left, right, threshold, features,right[node])
            print "}"
        else:
            print "return " + str(value[node])

    recurse(left, right, threshold, features, 0)


def top_k_features(estimator, top_k=None):
    """top_k features from a forest ensemble."""
    importances = estimator.feature_importances_
    sorted_features = np.argsort(importances)[::-1]
    if top_k is not None:
        sorted_features = sorted_features[:top_k]
    return sorted_features


def ova_forest_importance(X, cluster_labels, top_k=None):
    """Determine distinguishing cluster features based on
    RandomForest feature importances.
    """
    # fit a One-Vs-Rest classifier to distinguish clusters
    cluster_classifier = OneVsRestClassifier(
        estimator=RandomForestClassifier(n_estimators=500, n_jobs=-1))
    cluster_classifier.fit(X, cluster_labels)

    feature_importance = [top_k_features(estimator, top_k) for estimator in
                          cluster_classifier.estimators_]

    return feature_importance
