import numpy as np
from sklearn.base import clone

from . import (
    QuicGraphLasso,
    QuicGraphLassoCV,
    InverseCovarianceEstimator,
)


class AdaptiveGraphLasso(InverseCovarianceEstimator):
    """
    Two-stage adaptive estimator.

    a) Compute the first estimate with an estimator of your choice
       (self.estimator).  This estimate must produce a single precision_
       output.

    b) The resulting coefficients are used to generate adaptive weights and 
       the QuicGraphLasso is refit with these weights.

    See:
        "High dimensional covariance estimation based on Gaussian graphical
        models"
        S. Zhou, P. R{u}htimann, M. Xu, and P. B{u}hlmann
        ftp://ess.r-project.org/Manuscripts/buhlmann/gelato.pdf

    Add: "Relaxed lasso" by Meinshausen

    Parameters
    -----------        
    estimator : GraphLasso instance with model selection (default=QuicGraphLassoCV())
        After being fit, estimator.precision_ must either be a matrix.

    method : one of 'binary', 'glasso', 'inverse' (default='binary')
        binary: also called gelato or "relaxed lasso"
        glasso: 1/|coefficient|^2
        inverse: 1/|coefficient|

    Attributes
    ----------
    estimator_ : QuicGraphLasso instance
        The final estimator refit with adaptive weights.

    lam_ : 2D ndarray, shape (n_features, n_features)
        Adaptive weight matrix generated.
    """
    def __init__(self, estimator=None, method='binary'):
        self.estimator = estimator 
        self.method = method

        self.lam_ = None
        self.estimator_ = None

        # default to QuicGraphLassoCV
        if self.estimator is None:
            self.estimator = QuicGraphLassoCV()


    def _binary_weights(self, estimator):
        n_features, _ = estimator.precision_.shape
        lam = np.zeros((n_features, n_features))
        lam[estimator.precision_ == 0] = 1
        lam[np.diag_indices(n_features)] = 0
        return lam
        

    def _glasso_weights(self, estimator):
        n_features, _ = estimator.precision_.shape
        lam = np.zeros((n_features, n_features))
        mask = estimator.precision_ != 0
        lam[mask] = 1. / (np.abs(estimator.precision_[mask]) ** 2)
        mask_0 = estimator.precision_ == 0
        lam[mask_0] = np.max(lam.flat)
        lam[np.diag_indices(n_features)] = 0
        return lam


    def _inverse_weights(self, estimator):
        n_features, _ = estimator.precision_.shape
        lam = np.zeros((n_features, n_features))
        mask = estimator.precision_ != 0
        lam[mask] = 1. / np.abs(estimator.precision_[mask])
        mask_0 = estimator.precision_ == 0
        lam[mask_0] = np.max(lam.flat)
        lam[np.diag_indices(n_features)] = 0
        return lam


    def fit(self, X, y=None):
        """Estimate the precision using an adaptive maximum likelihood estimator.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the proportion matrix.
        """
        n_samples, n_features = X.shape

        # perform first estimate
        new_estimator = clone(self.estimator)
        new_estimator.fit(X)

        # generate weights
        if self.method == 'binary':
            self.lam_ = self._binary_weights(new_estimator)
        elif self.method == 'glasso':
            self.lam_ = self._glasso_weights(new_estimator)
        elif self.method == 'inverse':
            self.lam_ = self._inverse_weights(new_estimator)
        else:
            raise NotImplementedError(("Only method='binary', 'glasso', or",
                    "'inverse' have been implemented."))

        # perform second step adaptive estimate
        self.estimator_ = QuicGraphLasso(lam=self.lam_ * new_estimator.lam_, 
                                         mode='default',
                                         initialize_method='cov')
        self.estimator_.fit(X)

        self.is_fitted = True
        return self

