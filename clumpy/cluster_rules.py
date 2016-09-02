import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import _tree, DecisionTreeClassifier


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
        estimator=RandomForestClassifier(n_estimators=500, n_jobs=-1))
    cluster_classifier.fit(X, cluster_labels)

    feature_importance = [top_k_features(estimator, features=features, top_k=top_k) for estimator in
                          cluster_classifier.estimators_]

    return feature_importance


def train_decision_tree(X, cluster_labels, max_depth):
    decision_tree = OneVsRestClassifier(
            estimator=DecisionTreeClassifier(max_depth=max_depth, random_state=123))
    decision_tree.fit(X, cluster_labels)
    return decision_tree


def leave_paths(tree, class_name, feature_names=None):
    def recurse(tree, child_id, lineage=None):
        if lineage is None:
            values = tree.value[child_id][0, :]
            class_id = np.argmax(values)
            lineage = [[class_name[class_id], values[class_id]]]
        if child_id in tree.children_left:
            parent_id = np.where(tree.children_left == child_id)[0].item()
            split_type = '<='
        else:
            parent_id = np.where(tree.children_right == child_id)[0].item()
            split_type = '>'

        if feature_names is not None:
            feature = feature_names[tree.feature[parent_id]]
        else:
            feature = 'X[{0}]'.format(tree.feature[parent_id])
        threshold = round(tree.threshold[parent_id], 4)
        lineage.append((split_type, threshold, feature))

        if parent_id == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(tree, parent_id, lineage)

    leaf_nodes = np.argwhere(tree.children_left == _tree.TREE_LEAF)[:, 0]
    leaf_paths = [[node for node in recurse(tree, child_id)] for
                  child_id in leaf_nodes]

    return leaf_paths


def get_best_path(leaf_paths, class_name):
    leaves = [(idx, path[-1]) for idx, path in enumerate(leaf_paths) if
              path[-1][0] == class_name]
    best_path_idx = sorted(leaves, key=lambda x: x[1][1])[-1][0]
    return leaf_paths[best_path_idx]


def trim_path(leaf_path):
    rules = leaf_path[:-1]
    features_used = np.unique([rule[2] for rule in rules])
    description = ''
    for feature in features_used:
        feature_group = [rule for rule in rules if rule[2] == feature]
        feature_group = sorted(feature_group, key=lambda x: x[1])
        if len(feature_group) > 3:
            feature_group = [feature_group[0], feature_group[-1]]
        if len(feature_group) == 1:
            description += '{} {} {}'.format(
                    feature, feature_group[0][0], feature_group[0][1])
        else:
            description += '{} < {} <= {}'.format(
                    feature_group[0][1], feature, feature_group[1][1])

        description += ' AND\n'
    return description[:-4]


def tree_descriptions(X, cluster_labels, feature_names=None, max_depth=10):
    decision_tree = train_decision_tree(X, cluster_labels, max_depth=max_depth)

    leaf_descriptions = []
    for cluster_id, tree in enumerate(decision_tree.estimators_):
        cluster_name = 'cluster {}'.format(cluster_id)
        leaf_paths = leave_paths(
                tree.tree_,
                class_name=['not_cluster', cluster_name],
                feature_names=feature_names)
        best_path = get_best_path(leaf_paths, cluster_name)
        leaf_descriptions.append(trim_path(best_path))

    return leaf_descriptions


