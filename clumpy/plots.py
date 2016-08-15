import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

def plot_clusters(X, labels, centers):
    data = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'labels': labels})
    sns.lmplot('x', 'y', hue='labels', fit_reg=False, data=data)

    for center in centers:
        plt.plot(center[0], center[1], marker='o', color='black')
