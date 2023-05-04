import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, column_or_1d, check_consistent_length, check_scalar
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics.pairwise import rbf_kernel


class PWC(BaseEstimator, ClassifierMixin):
    """PWC
    The Parzen window classifier (PWC) [1] is a simple and probabilistic classifier. This classifier is based on a
    non-parametric density estimation obtained by applying a kernel function.

    Parameters
    ----------
    gamma: float, default=None
        Specifies the width of the RBF kernel. If None, defaults to 1.0 / n_features.
    alpha: int, default=1
        Prior counts of samples per class.

    Attributes
    ----------
    X_: numpy.ndarray, shape (n_samples, n_features)
        The sample matrix `X_` is the feature matrix representing the training samples.
    y_: array-like, shape (n_samples) or (n_samples, n_outputs)
        The array `y_` contains the class labels of the training samples.

    References
    ----------
    [1] O. Chapelle, "Active Learning for Parzen Window Classifier",
        Proceedings of the Tenth International Workshop Artificial Intelligence and Statistics, 2005.
    """

    def __init__(self, gamma=None, alpha=1.0):
        self.gamma = gamma
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the model using `X` as training data and `y` as class labels.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix `X` is the feature matrix representing the samples for training.
        y: array-like, shape (n_samples) or (n_samples, n_outputs)
            The array `y` contains the class labels of the training samples.

        Returns
        -------
        self: PWC,
            The `PWC` is fitted on the training data.
        """
        # Check attributes and parameters.
        if self.gamma is not None:
            check_scalar(self.gamma, min_val=0, target_type=float, name='gamma')
        check_scalar(self.alpha, min_val=0, target_type=float, name='alpha')
        self.X_ = check_array(X)
        self.y_ = column_or_1d(y)
        check_consistent_length(self.X_, self.y_)

        # Fit `LabelEncoder` object as `self.label_encoder_`.
        # TODO 

        # Transform `self.y_` using the fitted `self.label_encoder_`.
        # TODO 

        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X:  array-like, shape (n_samples, n_features)
            The sample matrix `X` is the feature matrix representing the training samples.

        Returns
        -------
        P:  array-like, shape (n_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered by lexicographic order.
        """

        # Calculate `n_test_samples x n_train_samples` kernel matrix `K`.
        # TODO 

        # TODO 

        # Calculate label frequency estimates as `n_test_samples x n_classes` matrix `F`
        # based on all training samples.
        # TODO 

        # Incorporate prior counts `alpha` into the kernel frequency matrix `F`.
        # TODO 

        # Normalize matrix `F` to obtain the probabilities `n_test_samples x n_features` matrix `P`.
        # TODO 

        return P

    def predict(self, X):
        """
        Return class label predictions for the test data X.

        Parameters
        ----------
        X:  array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y:  numpy.ndarray, shape = [n_samples]
            Predicted class labels class.
        """
        # Predict class labels `y`.
        # TODO 

        # Re-transform predicted labels using `self.label_encoder_`.
        # TODO 

        return y