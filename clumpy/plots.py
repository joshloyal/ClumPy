import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pandas.tools.plotting import parallel_coordinates

from clumpy import importance

sns.set_style('white')

CLUSTER_COLOR = sns.color_palette('Reds_d')[5]
MARGINAL_COLOR = sns.color_palette('Blues_d')[5]


def plot_clusters(X, labels=None, centers=None, size=10, aspect=2):
    if labels is None:
        labels = np.zeros(X.shape[0])
    data = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'labels': labels})

    p = sns.lmplot('x', 'y', hue='labels', fit_reg=False, data=data[labels != -1], size=size, aspect=aspect)

    outliers = data[labels == -1]
    plt.scatter(outliers.x.values, outliers.y.values, color='black', alpha=0.25)

    if centers is not None:
        for center in centers:
            plt.plot(center[0], center[1], marker='o', color='black')

    return p



def categorical_plot(y, cluster_labels, data, order='relevance', figsize=(10, 20)):
    cluster_ids = np.unique(cluster_labels)

    relevance = importance.categorical_relevance(y, cluster_labels, data)

    fig, axes = plt.subplots(len(cluster_ids), 1, sharex=True, figsize=figsize)
    for ax, cluster_id in zip(axes, cluster_ids):
        cluster_mask = (cluster_labels == cluster_id)
        sns.countplot(y=y, data=data, color=MARGINAL_COLOR, order=relevance[cluster_id], ax=ax)
        sns.countplot(y=y, data=data[cluster_mask], color=CLUSTER_COLOR, order=relevance[cluster_id], ax=ax)


def cat_plot(y, cluster_labels, cluster_id, data, ax=None):
    if ax is None:
        ax = plt.gca()

    relevance = importance.single_cluster_relevance(y, cluster_labels, cluster_id, data)
    cluster_mask = (cluster_labels == cluster_id)
    sns.countplot(y=y, data=data, color=MARGINAL_COLOR, order=relevance, ax=ax)
    sns.countplot(y=y, data=data[cluster_mask], color=CLUSTER_COLOR, order=relevance, ax=ax)



# this should be a parallel coordinate plot (means of cluster vs. marginal mean)
def quant_plot(y, cluster_labels, cluster_id, data, ax=None, scale=False):
    if ax is None:
        ax = plt.gca()

    width = 1
    cluster_mask = (cluster_labels == cluster_id)

    if scale:
        scaled_data = data.copy()
        #scaled_data[data.columns] = MinMaxScaler().fit_transform(data)
        scaled_data[y] = StandardScaler().fit_transform(data[y])
    else:
        scaled_data = data

    marginal_means = scaled_data[y].mean().to_frame()
    marginal_means = marginal_means.sort_values(by=0, ascending=False).transpose()
    columns = marginal_means.columns
    marginal_means['id'] = 'marginal'

    cluster_means = scaled_data[y][cluster_mask].mean().to_frame().transpose()
    cluster_means['id'] = 'cluster'.format(cluster_id)

    # sort in descending order of marginal means

    ax = parallel_coordinates(
            pd.concat((marginal_means, cluster_means)), 'id', cols=columns,
            color=[MARGINAL_COLOR, CLUSTER_COLOR],
            ax=ax)
    ax.get_yaxis().set_ticks([])
    plt.setp(ax.get_xticklabels(), rotation=45)
    #marginal_mean = np.mean(data[y])
    #cluster_mean = np.mean(data[y][cluster_mask])

    #if cluster_mean > marginal_mean:
    #    ax.barh(0.5, np.mean(data[y][cluster_mask]), width, color=CLUSTER_COLOR)
    #    ax.barh(0.5, np.mean(data[y]), width, color=MARGINAL_COLOR)
    #else:
    #    ax.barh(0.5, np.mean(data[y]), width, color=MARGINAL_COLOR)
    #    ax.barh(0.5, np.mean(data[y][cluster_mask]), width, color=CLUSTER_COLOR)
    #ax.get_yaxis().set_ticks([])
    #ax.set_xlabel(y)


def plot_cluster_statistics(cluster_labels, cluster_id, data, quant_var=[], qual_var=[], figsize=None, scale=False):
    fig, axes = plt.subplots(len(qual_var) + 1, 1, sharex=False, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, var in zip(axes[:-1], qual_var):
        cat_plot(var, cluster_labels, cluster_id, data, ax=ax)

    if quant_var:
        quant_plot(quant_var, cluster_labels, cluster_id, data, scale=scale, ax=axes[-1])
