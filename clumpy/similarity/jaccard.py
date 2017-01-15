from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np


def jaccard_similarity(set_a, set_b, assume_unique=True):
    """The Jaccard index, also known as the Jaccard similarity coefficient,
    is a statistic used to compare the similarity of sets. The index is
    defined as

    J(A, B) = |A \cup B| / |A \cap B| = |A \cap B| / (|A| + |B| - |A \cap B|).

    where J(A, B) = 1 if A and B are both empty.

    Parameters
    ----------
    set_a : array-like of shape [n_samples_b,]
        Elements in A

    set_b : array-like of shape [n_samples_a,]
        Elements in B

    assume_unique : bool (default=True)
        Whether to assume the inputs (set_a and set_b) only contain unique
        elements, i.e. are sets. If not then they are converted appropriatly.
    """
    if not assume_unique:
        set_a = np.unique(set_a)
        set_b = np.unique(set_b)

    n_elements_a = len(set_a)
    n_elements_b = len(set_b)
    n_total = n_elements_a + n_elements_b

    if n_total == 0:
        return 1.0
    else:
        n_intersect = len(np.intersect1d(set_a, set_b))
        return n_intersect / (n_total - n_intersect)
