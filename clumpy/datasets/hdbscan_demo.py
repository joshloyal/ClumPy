import os

import numpy as np

root = os.path.abspath(os.path.dirname(__file__))
DATA_NAME = 'data'
FILE_NAME = os.path.join(root, DATA_NAME, 'clusterable_data.npy')


def fetch_hdbscan_demo():
    data = np.load(FILE_NAME)
    return data
