import os
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold

root = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture
def soybean_data():
    df = pd.read_csv(os.path.join(root, 'test_data', 'soybeans.csv'))
    labels = df.pop('V35').values
    X = df.values
    X = VarianceThreshold().fit_transform(X)  # remove one category attributes
    return X, labels
