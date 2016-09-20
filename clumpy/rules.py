import numpy as np
import prim
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import _tree, DecisionTreeClassifier
from sklearn.preprocessing import Imputer

from clumpy import importance
from clumpy.preprocessing import process_data


def train_decision_tree(X, cluster_labels, max_depth, ova=False):
    """train_decision_tree

    Train a single decision tree to distinguish clusters.
    """
    if ova:
        decision_tree = OneVsRestClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=max_depth, random_state=123))
    else:
        decision_tree = DecisionTreeClassifier(
            max_depth=max_depth, random_state=123)
    decision_tree.fit(X, cluster_labels)
    return decision_tree


def leave_paths(tree, class_name, feature_names=None):
    """leave_paths

    Create a list of paths to each leaf node of a scikit-learn decision tree.

    Parameters
    ----------
    tree : decision tree classifier
        The decision tree to be broken down into decision paths. The
        tree is assumed to be trained in a binary classification fashion
        as of now.

    class_name : list of strings
        A list of the positive and negative class names, i.e.
        ['not cluster', 'cluster'].

    feature_names : list of strings, optional (default=None)
        Names of each of the features

    Returns
    -------
    leaf_paths : list of list of tuples
        A list of the leaf paths. A typical leaf path will look like
        [('<=', 0.8, 'x0'), ('>', 0.2, 'x1'), ('cluster', 1234)]
        where the last element corresponds to the class of that data
        partition and the number of samples of that class in the
        training dataset.
    """
    if not isinstance(tree, _tree.Tree):
        tree = tree.tree_

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
    """get_best_path

    Determine the best path to use as the description from all the
    paths in a decision tree. This is simply chosen as the partition
    with the most samples (may be good to have a threshold of leaf purity)

    We could do an OR on the different partitions.
    """
    leaves = [(idx, path[-1]) for idx, path in enumerate(leaf_paths) if
              path[-1][0] == class_name]
    best_path_idx = sorted(leaves, key=lambda x: x[1][1])[-1][0]
    return leaf_paths[best_path_idx]


def trim_path(leaf_path, categorical_columns=[]):
    """trim_path

    Trim the path and turn it into a human readable string. This
    will combine multiple cuts on the same variable by taking maximums
    when (<=) and minimums when (>).
    """
    rules = leaf_path[:-1]
    features_used = np.unique([rule[2] for rule in rules])
    description = ''
    for feature in features_used:
        feature_group = [rule for rule in rules if rule[2] == feature]
        feature_group = sorted(feature_group, key=lambda x: x[1])
        if len(feature_group) > 3:
            feature_group = [feature_group[0], feature_group[-1]]
        if len(feature_group) == 1:
            op = feature_group[0][0]
            feature_val = feature_group[0][1]
            if feature in categorical_columns:
                feature, feature_val = feature.rsplit('_', 1)
                if op == '<=':
                    op = '!='
                else:
                    op = '=='
            else:
                op = feature_group[0][0]

            description += '{} {} {}'.format(
                    feature, op, feature_val)
        else:
            description += '{} < {} <= {}'.format(
                    feature_group[0][1], feature, feature_group[1][1])

        description += ' AND\n'
    return description[:-4]


def tree_descriptions(data, cluster_labels,
                      categorical_columns=[], feature_names=None,
                      max_depth=10):
    """tree_descriptions

    Determine 'human readable' descriptions for clusters using the rules
    of a decision tree algorithm. Specifically, a multi-class decision
    tree is fit in a one-vs-all fashion to the clusters of the dataset.
    Then the decision path that contains the partition with the largest
    number of members of the cluster is chosen as the rules to display.
    This decision path is extracted and turned into a human readable sentence
    for interpretation by the user.

    Parameters
    ----------
    X : pd.DataFrame of shape [n_samples, n_features]
        Data array to fit the decision tree algorithm
    cluster_labels : array-like of shape [n_samples,]
        The labels [0 - n_classes] of the corresponding clusters
    feature_names : dict of lists of strings (optional)
        The names of each feature in the dataset for a particular cluster
    max_depth : int, optional (default=5)
        Depth of the decision tree. This controls how many rules
        are generated.

    Returns
    -------
    leaf_descriptions : list of strings
        The descriptions of each cluster. The position in the list
        corresponds to the cluster_id.
    """
    leaf_descriptions = []
    for cluster_id in np.unique(cluster_labels):
        cluster_features = feature_names[cluster_id]
        X, num_features, cat_features = process_data(
                data[cluster_features],
                categorical_columns=[var for var in cluster_features if var in categorical_columns],
                cat_preprocessing='onehot')

        tree = train_decision_tree(
                X, cluster_labels == cluster_id, max_depth=max_depth, ova=False)

        cluster_name = 'cluster {}'.format(cluster_id)
        leaf_paths = leave_paths(
                tree,
                class_name=['not_cluster', cluster_name],
                feature_names=num_features + cat_features)
        best_path = get_best_path(leaf_paths, cluster_name)
        leaf_descriptions.append(trim_path(best_path, categorical_columns=cat_features))

    return leaf_descriptions

def prim_descriptions(data, cluster_labels, feature_names=[]):
    boxes = []
    for cluster_id in np.unique(cluster_labels):
        cluster_features = feature_names[cluster_id]
        p = prim.Prim(data,
                      cluster_labels == cluster_id,
                      threshold=0.5,
                      threshold_type='>',
                      include=cluster_features)
        boxes.append(p.find_box())
    return boxes
