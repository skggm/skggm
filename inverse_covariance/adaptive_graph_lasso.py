from __future__ import absolute_import

import numpy as np
from sklearn.utils import check_array, as_float_array, deprecated
from sklearn.base import BaseEstimator

from . import QuicGraphicalLasso, QuicGraphicalLassoCV, InverseCovarianceEstimator


class AdaptiveGraphicalLasso(BaseEstimator):
    """
    Two-stage adaptive estimator.

    a) Compute the first estimate with an estimator of your choice
       (self.estimator).  This estimate must produce a single precision_
       output.

    b) The resulting coefficients are used to generate adaptive weights and
       the QuicGraphicalLasso is refit with these weights.

    See:
        "High dimensional covariance estimation based on Gaussian graphical
        models"
        S. Zhou, P. R{\"u}htimann, M. Xu, and P. B{\"u}hlmann

        "Relaxed Lasso"
        N. Meinshausen, December 2006.

    Parameters
    -----------
    estimator : GraphicalLasso instance with model selection
                (default=QuicGraphicalLassoCV())
        After being fit, estimator.precision_ must either be a matrix.

    method : one of 'binary', 'inverse_squared', 'inverse' (default='binary')
        binary: non-zero where estimator was zero, 1 else
                (gelato, "relaxed lasso")
        inverse_squared: 1/|coefficient|^2  (glasso)
        inverse: 1/|coefficient|

    Attributes
    ----------
    estimator_ : QuicGraphicalLasso instance
        The final estimator refit with adaptive weights.

    lam_ : 2D ndarray, shape (n_features, n_features)
        Adaptive weight matrix generated.
    """

    def __init__(self, estimator=None, method="binary"):
        self.estimator = estimator
        self.method = method

    def _binary_weights(self, estimator):
        n_features, _ = estimator.precision_.shape
        lam = np.zeros((n_features, n_features))
        lam[estimator.precision_ == 0] = 1
        lam[np.diag_indices(n_features)] = 0
        return lam

    def _inverse_squared_weights(self, estimator):
        n_features, _ = estimator.precision_.shape
        lam = np.zeros((n_features, n_features))
        mask = estimator.precision_ != 0
        lam[mask] = 1. / (np.abs(estimator.precision_[mask]) ** 2)
        mask_0 = estimator.precision_ == 0
        lam[mask_0] = np.max(lam[mask].flat)  # non-zero in appropriate range
        lam[np.diag_indices(n_features)] = 0
        return lam

    def _inverse_weights(self, estimator):
        n_features, _ = estimator.precision_.shape
        lam = np.zeros((n_features, n_features))
        mask = estimator.precision_ != 0
        lam[mask] = 1. / np.abs(estimator.precision_[mask])
        mask_0 = estimator.precision_ == 0
        lam[mask_0] = np.max(lam[mask].flat)  # non-zero in appropriate range
        lam[np.diag_indices(n_features)] = 0
        return lam

    def fit(self, X, y=None):
        """Estimate the precision using an adaptive maximum likelihood estimator.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the proportion matrix.
        """
        # default to QuicGraphicalLassoCV
        estimator = self.estimator or QuicGraphicalLassoCV()

        self.lam_ = None
        self.estimator_ = None

        X = check_array(X, ensure_min_features=2, estimator=self)
        X = as_float_array(X, copy=False, force_all_finite=False)

        n_samples_, n_features_ = X.shape

        # perform first estimate
        estimator.fit(X)

        if self.method == "binary":
            # generate weights
            self.lam_ = self._binary_weights(estimator)

            # perform second step adaptive estimate
            self.estimator_ = QuicGraphicalLasso(
                lam=self.lam_ * estimator.lam_,
                mode="default",
                init_method="cov",
                auto_scale=False,
            )
            self.estimator_.fit(X)

        elif self.method == "inverse_squared":
            self.lam_ = self._inverse_squared_weights(estimator)

            # perform second step adaptive estimate
            self.estimator_ = QuicGraphicalLassoCV(
                lam=self.lam_ * self.estimator.lam_, auto_scale=False
            )
            self.estimator_.fit(X)

        elif self.method == "inverse":
            self.lam_ = self._inverse_weights(estimator)

            # perform second step adaptive estimate
            self.estimator_ = QuicGraphicalLassoCV(
                lam=self.lam_ * estimator.lam_, auto_scale=False
            )
            self.estimator_.fit(X)

        else:
            raise NotImplementedError(
                (
                    "Only method='binary', 'inverse_squared', or",
                    "'inverse' have been implemented.",
                )
            )

        self.is_fitted_ = True
        return self


@deprecated(
    "The class AdaptiveGraphLasso is deprecated "
    "Use class AdaptiveGraphicalLasso instead."
)
class AdaptiveGraphLasso(AdaptiveGraphicalLasso):
    pass
