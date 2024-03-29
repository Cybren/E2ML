"""
This code implement a full factorial approach.
"""

import numpy as np


def full_fac(levels):
    """
    Create a general full-factorial design

    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.

    Returns
    -------
    X : np.ndarray of shape (n_combs, n_factors)
        The design matrix with coded levels 0 to k-1 for a k-level factor

    Example
    -------
    ::

        >>> fullfact([2, 4, 3])
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  0.],
               [ 0.,  2.,  0.],
               [ 1.,  2.,  0.],
               [ 0.,  3.,  0.],
               [ 1.,  3.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 0.,  2.,  1.],
               [ 1.,  2.,  1.],
               [ 0.,  3.,  1.],
               [ 1.,  3.,  1.],
               [ 0.,  0.,  2.],
               [ 1.,  0.,  2.],
               [ 0.,  1.,  2.],
               [ 1.,  1.,  2.],
               [ 0.,  2.,  2.],
               [ 1.,  2.,  2.],
               [ 0.,  3.,  2.],
               [ 1.,  3.,  2.]])
    """
    levels = np.array(levels)
    prod = levels.prod()
    X = np.zeros((prod, len(levels)))
    for i in range(prod):
        mod = 1
        for j in range(len(levels)):
            X[i, j] = i // mod % levels[j]
            mod *= levels[j]
    return X.astype(int)
