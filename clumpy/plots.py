import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

def plot_clusters(X, labels, centers=None):
    data = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'labels': labels})
    sns.lmplot('x', 'y', hue='labels', fit_reg=False, data=data[labels != -1])

    outliers = data[labels == -1]
    plt.scatter(outliers.x.values, outliers.y.values, color='black', alpha=0.25)

    if centers is not None:
        for center in centers:
            plt.plot(center[0], center[1], marker='o', color='black')
