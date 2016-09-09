import numpy as np
from sklearn.utils import check_random_state


from clumpy.cluster.k_modes import KModes
from clumpy.preprocessing import OrdinalEncoder


def gen_data(n_samples=100, random_state=1):
    random_state = check_random_state(random_state)

    x1 = random_state.choice(np.arange(2), n_samples, p=[0.2, 0.8]).reshape(-1, 1)
    x2 = random_state.choice(np.arange(10), n_samples).reshape(-1, 1)
    x3 = random_state.choice(np.arange(3), n_samples, p=[0.1, 0.2, 0.7]).reshape(-1, 1)

    return np.hstack((x1, x2, x3))


def test_k_modes():
    X = gen_data(n_samples=10000)
    kmodes = KModes(n_clusters=8, verbose=True, n_init=10)
    kmodes.fit(X)
