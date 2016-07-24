import numpy as np
from sklearn.base import BaseEstimator 

import metrics


def _initialize_coefficients(X, method='corrcoef'):
    if method is 'corrcoef':
        return np.corrcoef(X, rowvar=False), 1.0
    elif method is 'cov':   
        init_cov = np.cov(X, rowvar=False)
        return init_cov, np.max(np.abs(np.triu(init_cov)))
    else:
        raise ValueError("initialize_method must be 'corrcoef' or 'cov'.")


def _compute_error(comp_cov, covariance_, precision_, score_metric='frobenius'):
    """Computes the covariance error vs. comp_cov.
        
    Parameters
    ----------
    comp_cov : array-like, shape = (n_features, n_features)
        The precision to compare with.
        This should normally be the test sample covariance/precision.
            
    scaling : bool
        If True, the squared error norm is divided by n_features.
        If False (default), the squared error norm is not rescaled.

    score_metric : str
        The type of norm used to compute the error between the estimated 
        self.precision, self.covariance and the reference `comp_cov`. 
        Available error types:
        
        - 'frobenius' (default): sqrt(tr(A^t.A))
        - 'spectral': sqrt(max(eigenvalues(A^t.A))
        - 'kl': kl-divergence 
        - 'quadratic': quadratic loss
        - 'log_likelihood': negative log likelihood
    
    squared : bool
        Whether to compute the squared error norm or the error norm.
        If True (default), the squared error norm is returned.
        If False, the error norm is returned.
    """
    if score_metric == "frobenius":
        error = comp_cov - covariance_
        return np.sum(error ** 2)                        
    elif score_metric == "spectral":
        error = comp_cov - covariance_
        return np.amax(np.linalg.svdvals(np.dot(error.T, error)))
    elif score_metric == "kl":
        return metrics.kl_loss(comp_cov, precision_)
    elif score_metric == "quadratic":
        return metrics.quadratic_loss(comp_cov, precision_)
    elif score_metric == "log_likelihood":
        return -metrics.log_likelihood(comp_cov, precision_)
    else:
        raise NotImplementedError(("Must be frobenius, spectral, kl, "
                                   "quadratic, or log_likelihood"))
    
class InverseCovarianceEstimator(BaseEstimator):
    """
    Base class for inverse covariance estimators.

    Provides initialization method, metrics, scoring function, 
    and ebic model selection.

    Parameters
    -----------        
    lam : scalar or 2D ndarray, shape (n_features, n_features) (default=0.5)
        Regularization parameters per element of the inverse covariance matrix.
    
    mode : one of 'default', 'path', or 'trace'
        Computation mode.

    path : array of floats (default=None)
        In "path" mode, an array of float values for scaling lam.
        The path must be sorted largest to smallest.  This class will auto sort
        this, in which case indices correspond to self.path_

    score_metric : one of 'log_likelihood' (default), 'frobenius', 'spectral',
                  'kl', or 'quadratic'
        Used for computing self.score().

    initialize_method : one of 'corrcoef', 'cov'
        Computes initial covariance and scales lambda appropriately.

    Attributes
    ----------
    covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated covariance matrix
        
        If mode='path', this is a len(path) list of
        2D ndarray, shape (n_features, n_features)


    precision_ : 2D ndarray, shape (n_features, n_features)
        Estimated pseudo-inverse matrix.
        
        If mode='path', this is a len(path) list of
        2D ndarray, shape (n_features, n_features)

    sample_covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated sample covariance matrix

    lam_scale_ : (float)
        Additional scaling factor on lambda (due to magnitude of 
        sample_covariance_ values).

    path_ : None or array of floats
        Sorted (largest to smallest) path.  This will be None if not in path
        mode.
    """
    def __init__(self, lam=0.5, mode='default', score_metric='log_likelihood',
                 path=None, initialize_method='corrcoef'):
        self.lam = lam
        self.mode = mode
        self.score_metric = score_metric
        self.initialize_method = initialize_method
        self.set_path(path)

        self.covariance_ = None
        self.precision_ = None

        # these must be updated upon self.fit()
        self.sample_covariance_ = None
        self.lam_scale_ = None
        self.n_samples = None
        self.n_features = None
        self.is_fitted = False 

        super(InverseCovarianceEstimator, self).__init__()


    def lam_select_(self, lam_index):
        return self.lam * self.lam_scale_ * self.path[lam_index]


    def initialize_coefficients(self, X):
        """Computes ...

        Initialize the following values:
            self.n_samples
            self.n_features
            self.sample_covariance_
            self.lam_scale_
        """
        self.n_samples, self.n_features = X.shape
        self.sample_covariance_, self.lam_scale_ = _initialize_coefficients(
                X,
                method=self.initialize_method)

    def set_path(self, path):
        """Sorts path values from largest to smallest.

        Will warn if path parameter was not already sorted.
        """
        if self.mode is 'path' and path is None:
            raise ValueError("path required in path mode.")
            return

        if path is None:
            self.path = None
            return

        self.path = np.array(sorted(set(path), reverse=True))
        if self.path[0] != path[0]:
            print 'Warning: Path must be sorted largest to smallest.'


    def score(self, X_test, y=None):
        """Computes the score between cov/prec of sample covariance of X_test 
        and X via 'score_metric'.

        Note: We want to maximize score so we return the negative error 
              (or the max negative error). 
       
        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).
        
        y : not used.
        
        Returns
        -------
        result : float
            #The likelihood of the data set with `self.covariance_` as an
            #estimator of its covariance matrix.
        """        
        if self.mode is 'path':
            raise NotImplementedError(("self.score is not implemented for path "
                                        "mode.  Use QuicGraphLassoCV."))

        S_test, lam_scale_test = _initialize_coefficients(
                X_test,
                method=self.initialize_method)
        error = self.cov_error(S_test, score_metric=self.score_metric)

        # maximize score with -error
        return -error


    def cov_error(self, comp_cov, score_metric='frobenius'):
        """Computes the covariance error vs. comp_cov.
        
        Parameters
        ----------
        comp_cov : array-like, shape = (n_features, n_features)
            The precision to compare with.
            This should normally be the test sample covariance/precision.
                
        scaling : bool
            If True, the squared error norm is divided by n_features.
            If False (default), the squared error norm is not rescaled.

        score_metric : str
            The type of norm used to compute the error between the estimated 
            self.precision, self.covariance and the reference `comp_cov`. 
            Available error types:
            
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            - 'kl': kl-divergence 
            - 'quadratic': quadratic loss
            - 'log_likelihood': negative log likelihood
        
        squared : bool
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.
        
        Returns
        -------
        The min error between `self.covariance_` and `comp_cov` 
        """  
        if self.mode is not 'path':
            return _compute_error(comp_cov,
                                self.covariance_,
                                self.precision_,
                                score_metric)

        path_errors = []
        for lidx, lam in enumerate(self.path):
            path_errors.append(_compute_error(comp_cov,
                                            self.covariance_[lidx],
                                            self.precision_[lidx],
                                            score_metric))

        return path_errors


    def ebic_select(self, gamma=0):
        '''
        Uses Extended Bayesian Information Criteria for model selection.

        Can only be used in path mode.

        See:
            Extended Bayesian Information Criteria for Gaussian Graphical Models
            R. Foygel and M. Drton
            NIPS 2010

        Parameters
        ----------
        gamma : (float) \in (0, 1)
            Choice of gamma=0 leads to classical BIC
            Positive gamma leads to stronger penalization of large graphs.

        Returns
        -------
        Lambda index with best ebic score.
        '''
        # must be path mode
        if self.mode is not 'path':
            raise NotImplementedError(("Model selection only implemented for " 
                                      "mode=path"))
            return

        # model must be fitted
        if not self.is_fitted:
            return

        ebic_scores = []
        for lidx, lam in enumerate(self.path):
            ebic_scores.append(metrics.ebic(
                    self.sample_covariance_,
                    self.precision_[lidx],
                    self.n_samples,
                    self.n_features,
                    gamma=gamma))

        return np.argmin(ebic_scores)

    