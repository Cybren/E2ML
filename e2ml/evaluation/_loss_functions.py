import numpy as np
from scipy.special import xlogy
def zero_one_loss(y_true, y_pred):
    """
    Computes the empirical risk for the zero-one loss function.

    Parameters
    ----------
    y_true : array-like of shape (n_labels,)
        True class labels as array-like object.
    y_pred : array-like of shape (n_labels,)
        Predicted class labels as array-like object.

    Returns
    -------
    risk : float in [0, 1]
        Empirical risk computed via the zero-one loss function.
    """
    if(len(y_true) != len(y_pred)):
        print("arrays do not have the same length!")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.abs(y_true-y_pred))/len(y_true)

def binary_cross_entropy_loss(y_true, y_pred):
    """
    Computes the empirical risk for the binary cross entropy (BCE) loss function.

    Parameters
    ----------
    y_true : array-like of shape (n_labels,)
        True conditional class probabilities as array-like object.
    y_pred : array-like of shape (n_labels,)
        Predicted conditional class probabilities as array-like object.

    Returns
    -------
    risk : float in [0, +infinity]
        Empirical risk computed via the BCE loss function.
    """
    if(len(y_true) != len(y_pred)):
        print("arrays do not have the same length!")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check value ranges of probabilities and raise ValueError if the ranges are invalid. In this case, it should be
    # allowed to have estimated probabilities in the interval [0, 1] instead of only (0, 1).
    # TODO 

    if(not (np.all(y_true >= 0) and np.all(y_true <=1) and np.all(y_pred >= 0) and np.all(y_pred <=1))):
        raise ValueError("asdf")
        
    return np.mean(- xlogy(y_true, y_pred) - xlogy((1-y_true), (1-y_pred)))
    #return np.sum(- y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))/len(y_true)