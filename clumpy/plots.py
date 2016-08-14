import pandas as pd
import seaborn as sns


def plot_clusters(X, labels):
    data = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'labels': labels})
    return sns.lmplot('x', 'y', hue='labels', fit_reg=False, data=data)
