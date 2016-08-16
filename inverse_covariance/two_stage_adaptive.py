import numpy as np
from sklearn.base import BaseEstimator 
from sklearn.utils.extmath import fast_logdet

from inverse_covariance import _initialize_coefficients

class TwoStageAdaptive(BaseEstimator):
    """
    Two-stage adaptive meta-estimator.

    This meta-estimator computes an initial estimate using QuicGraphLasso,
    QuicGraphLassoCV (should also be compatible with scikit-learn GraphLasso).

    Using the resulting coefficients (adaptive) this estimator weights the sample
    covariance when computing a maximum likelihood estimate of the graph.

    # References here

    Parameters
    -----------        
    estimator : 
        After being fit, estimator.precision_ must either be a matrix with the 
        precision or a list of precision matrices (e.g., path mode).
        This should be compatible with QuicGraphLasso, QuicGraphLassoCV, as well
        as the scikit-learn variants.

    estimator_args :

    method : one of 'gelato', 'glasso', 'inverse'
        gelato: binary
        glasso: 1/|coefficient|^2
        inverse: 1/|coefficient|


    Attributes
    ----------
    precision_ 

    estimator_
    
    """
    def __init__(self, estimator=None, estimator_args={}, method='gelato'):
        self.estimator = estimator 
        self.estimator_args = estimator_args
        self.method = method

        self.precision_ = None
        self.estimator_ = None
        self.sample_covariance_ = None

    def _gelato(self):
        n_features, _ = self.estimator_.precision_.shape
        theta = np.zeros((n_features, n_features))
        theta[self.estimator_.precision_ != 0] = 1
        theta_x_s = np.dot(theta, self.sample_covariance_)
        return np.trace(theta_x_s) - fast_logdet(theta) # if theta is sparse, is this just sparsity?

    def _glasso(self):
        n_features, _ = self.estimator_.precision_.shape
        theta = 1./(np.abs(self.estimator_.precision_) ** 2)
        theta_x_s = np.dot(theta, self.sample_covariance_)
        return np.trace(theta_x_s) - fast_logdet(theta)

    def _inverse(self):
        n_features, _ = self.estimator_.precision_.shape
        theta = 1./(np.abs(self.estimator_.precision_) ** 2)
        theta_x_s = np.dot(theta, self.sample_covariance_)
        return np.trace(theta_x_s) - fast_logdet(theta) 

    def fit(self, X, y=None):
        """Estimate the precision using an adaptive maximum likelihood estimator.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the proportion matrix.
        """
        n_samples, n_features = X.shape

        # perform first estimate
        self.estimator_ = self.estimator(**self.estimator_args)
        self.estimator_.fit(X)

        # get covariance with 'cov'
        self.sample_covariance_, lam_scale = _initialize_coefficients(
            X,
            method='cov')

        # helper functions such as self._gelato() return a scalar and we want to
        # return the argmin.  Does this mean we need multiple precisions?
        # Oh I see, those should be like scores for cross validation...

        # estimate MLE precision with estimator weights
        #if self.method == 'gelato':
        #    self.precision_ = self._gelato()
        #elif self.method == 'glasso':
        #    self.precision_ = self._glasso()
        #elif self.method == 'inverse':
        #    self.precision_ = self._inverse()
        #else:
        #    raise NotImplementedError(("Only method='gelato', 'glasso', or",
        #            "'inverse' have been implemented."))

        

        


