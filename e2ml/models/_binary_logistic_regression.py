import numpy as np

from e2ml.evaluation import binary_cross_entropy_loss

from scipy.special import expit
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, column_or_1d, check_consistent_length, check_scalar
from sklearn.preprocessing import LabelEncoder


class BinaryLogisticRegression(BaseEstimator, ClassifierMixin):
    """BinaryLogisticRegression

    Binary logistic regression (BLR) is a simple probabilistic classifier for binary classification problems.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of optimization steps.
    lmbda: float, default=0.0
        Regularization hyperparameter.


    Attributes
    ----------
    w_: numpy.ndarray, shape (n_features,)
        Weights (parameters) optimized during training the BLR model.
    """

    def __init__(self, maxiter=100, lmbda=0.0):
        self.maxiter = maxiter
        self.lmbda = lmbda

    def fit(self, X, y):
        """
        Fit the `BinaryLogisticRegression` model using `X` as training data and `y` as class labels.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix `X` is the feature matrix representing the samples for training.
        y: array-like, shape (n_samples) or (n_samples, n_outputs)
            The array `y` contains the class labels of the training samples.

        Returns
        -------
        self: BinaryLogisticRegression,
            The `BinaryLogisticRegression` model fitted on the training data.
        """
        # Check attributes.
        check_scalar(self.maxiter, min_val=0, name='maxiter', target_type=int)
        check_scalar(self.lmbda, min_val=0, name='lmbda', target_type=(int, float))
        X = check_array(X)
        self._check_n_features(X, reset=True)
        y = column_or_1d(y)
        check_consistent_length(X, y)

        # Fit `LabelEncoder` object as `self.label_encoder_`.
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y)

        # Raise `ValueError` if there are more than two classes.
        if(len(self.label_encoder_.classes_) > 2):
            raise ValueError("asdf")

        # Transform `self.y_` using the fitted `self.label_encoder_`.
        self.y_ = self.label_encoder_.transform(y) 

        # Initialize weights `w0`.
        w0 = np.zeros()

        def loss_func(w):
            """
            Compute the (scaled) loss with respect to weights `w`.

            Parameters
            ----------
            w : np.ndarray of shape (n_features,)

            Returns
            -------
            loss : float
                Evaluated (scaled) loss.
            """
            # Compute predictions for given weights.
            y_pred = expit(np.matmul(X, w))

            # Compute binary cross entropy loss including regularization.
            loss = binary_cross_entropy_loss(y_true=y, y_pred=y_pred)
            loss += 0.5 * len(X)**(-1) * self.lmbda * w.T @ w

            return loss

        def gradient_func(w):
            # Compute predictions for given weights.
            y_pred = expit(np.matmul(X, w))

            # Compute gradient.
            gradient = len(X)**(-1) * np.sum((y_pred - y)[:, None] * X)
            gradient += self.lmbda * w
            return gradient

        # Use `scipy.optimize.minimize` with `BFGS` as `method` to optimize the loss function and store the result as
        # `self.w_`
        self.w_ = minimize(fun=loss_func, x0=w0, jac=gradient_func, method='BFGS', options={"maxiter" : self.maxiter}).x

        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test data `X`.

        Parameters
        ----------
        X:  array-like of shape (n_samples, n_features)
            The sample matrix `X` is the feature matrix representing the training samples.

        Returns
        -------
        P:  numpy.ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered by lexicographic order.
        """
        # Check `X` parameter.
        X = check_array(X)
        self._check_n_features(X, reset=False)

        # Estimate and return conditional class probabilities.
        y_pred = expit(np.matmul(X,self.w_))
        #return y_pred.reshape(X.shape[0], X.shape[1])
        return np.column_stack((1-y_pred, y_pred))

    def predict(self, X):
        """
        Return class label predictions for the test data `X`.

        Parameters
        ----------
        X:  array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y:  numpy.ndarray of shape = [n_samples]
            Predicted class labels class.
        """
        # Predict class labels `y`.
        y = self.predict_proba(X).argmax(axis=1)

        # Re-transform predicted labels using `self.label_encoder_`.
        y = self.label_encoder_.inverse_transform(y)

        return y