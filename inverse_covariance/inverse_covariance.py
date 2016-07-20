import numpy as np
from sklearn.base import BaseEstimator 
from sklearn.utils import check_array, as_float_array
from sklearn.utils.extmath import fast_logdet, pinvh

import pyquic



def log_likelihood(covariance, precision):
    """Computes ...
    
    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance
    
    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested
    
    Returns
    -------
    log-likelihood
    """
    assert covariance.shape == precision.shape
    dim, _ = precision.shape
    log_likelihood_ = - np.trace(np.dot(covariance, precision)) + fast_logdet(precision)
    log_likelihood_ -= dim * np.log(2 * np.pi)
    log_likelihood_ /= 2.
    return log_likelihood_


def kl_loss(covariance, precision):
    """Computes the KL divergence between precision estimate and 
    reference covariance.
    
    The loss is computed as:

        Trace(Theta_1 * Sigma_0) - log(Theta_0 * Sigma_1) - dim(Sigma)

    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance
    
    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested
    
    Returns
    -------
    KL-divergence 
    """
    assert covariance.shape == precision.shape
    dim, _ = precision.shape
    p_dot_c = np.dot(precision, covariance)
    return 0.5 * (np.trace(p_dot_c) - fast_logdet(p_dot_c) - dim)


def quadratic_loss(covariance, precision):
    """Computes ...
    
    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance
    
    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the model to be tested
    
    Returns
    -------
    Quadratic loss
    """
    assert covariance.shape == precision.shape
    dim, _ = precision.shape
    return np.trace((covariance * precision - np.eye(dim))**2)


def ebic(covariance, precision, n_samples, n_features, gamma=0):
    '''
    Extended Bayesian Information Criteria for model selection.

    When using path mode, use this as an alternative to cross-validation for
    finding lambda.

    See:
        Extended Bayesian Information Criteria for Gaussian Graphical Models
        R. Foygel and M. Drton
        NIPS 2010

    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance (sample covariance)
    
    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the model to be tested

    gamma : (float) \in (0, 1)
        Choice of gamma=0 leads to classical BIC
        Positive gamma leads to stronger penalization of large graphs.

    Returns
    -------
    ebic score (float).  Caller should minimized this score.
    '''
    l_theta = log_likelihood(covariance, precision)
    precision_t = np.empty(precision.shape)
    precision_t[:] = precision
    precision_t[precision_t < 1e-1] = 0
    precision_nnz = np.count_nonzero(precision_t)
    return -2.0 * l_theta +\
            precision_nnz * np.log(n_samples) +\
            4.0 * precision_nnz * np.log(n_features) * gamma
    

def quic(S, lam, mode='default', tol=1e-6, max_iter=1000, 
        Theta0=None, Sigma0=None, path=None, msg=0):
    """Fits the inverse covariance model according to the given training 
    data and parameters.

    Parameters
    -----------
    S : 2D ndarray, shape (n_features, n_features)
        Empirical covariance or correlation matrix.

    Other parameters described in `class InverseCovariance`.

    Returns
    -------
    Theta : 
    Sigma : 
    opt : 
    cputime : 
    iters : 
    dGap : 
    """
    assert mode in ['default', 'path', 'trace'],\
            'mode = \'default\', \'path\' or \'trace\'.'

    Sn, Sm = S.shape
    if Sn != Sm:
        raise ValueError("Input data must be square. S shape = {}".format(
                         S.shape))
        return

    # Regularization parameter matrix L.
    if isinstance(lam, float):
        _lam = np.empty((Sn, Sm))
        _lam[:] = lam
    else:
        assert L.shape == S.shape, 'lam, S shape mismatch.'
        _lam = as_float_array(lam, copy=False, force_all_finite=False)
 
    # Defaults.
    optSize = 1
    iterSize = 1
    if mode is "trace":
        optSize = max_iter

    # Default Theta0, Sigma0 when both are None.
    if Theta0 is None and Sigma0 is None:
        Theta0 = np.eye(Sn)
        Sigma0 = np.eye(Sn)

    assert Theta0 is not None,\
            'Theta0 and Sigma0 must both be None or both specified.'
    assert Sigma0 is not None,\
            'Theta0 and Sigma0 must both be None or both specified.'
    assert Theta0.shape == S.shape, 'Theta0, S shape mismatch.'
    assert Sigma0.shape == S.shape, 'Theta0, Sigma0 shape mismatch.'
    Theta0 = as_float_array(Theta0, copy=False, force_all_finite=False)
    Sigma0 = as_float_array(Sigma0, copy=False, force_all_finite=False)

    if mode is 'path':
        assert path is not None, 'Please specify the path scaling values.'
        path_len = len(path)
        optSize = path_len
        iterSize = path_len

        # Note here: memory layout is important:
        # a row of X/W holds a flattened Sn x Sn matrix,
        # one row for every element in _path_.
        Theta = np.empty((path_len, Sn * Sn))
        Theta[0,:] = Theta0.ravel()
        Sigma = np.empty((path_len, Sn * Sn))
        Sigma[0,:] = Sigma0.ravel()
    else:
        path = np.empty(1)
        path_len = len(path)

        Theta = np.empty(Theta0.shape)
        Theta[:] = Theta0
        Sigma = np.empty(Sigma0.shape)
        Sigma[:] = Sigma0
                    
    # Run QUIC.
    opt = np.zeros(optSize)
    cputime = np.zeros(optSize)
    dGap = np.zeros(optSize)
    iters = np.zeros(iterSize, dtype=np.uint32)
    pyquic.quic(mode, Sn, S, _lam, path_len, path, tol, msg, max_iter,
                Theta, Sigma, opt, cputime, iters, dGap)

    if optSize == 1:
        opt = opt[0]
        cputime = cputime[0]
        dGap = dGap[0]

    if iterSize == 1:
        iters = iters[0]

    return Theta, Sigma, opt, cputime, iters, dGap


class InverseCovariance(BaseEstimator):
    """
    Computes a sparse inverse covariance matrix estimation using quadratic 
    approximation. 

    The inverse covariance is estimated the sample covariance estimate 
    $S$ as an input such that: 

    $T_hat = max_{\Theta} logdet(Theta) - Trace(ThetaS) - \lambda|\Theta|_1 $

    Parameters
    -----------        
    lam : scalar or 2D ndarray, shape (n_features, n_features) (default=0.5)
        Regularization parameters per element of the inverse covariance matrix.
    
    mode : one of 'default', 'path', or 'trace'
        Computation mode.

    tol : float (default=1e-6)
        Convergence threshold.

    max_iter : int (default=1000)
        Maximum number of Newton iterations.

    Theta0 : 2D ndarray, shape (n_features, n_features) (default=None) 
        Initial guess for the inverse covariance matrix. If not provided, the 
        diagonal identity matrix is used.

    Sigma0 : 2D ndarray, shape (n_features, n_features) (default=None)
        Initial guess for the covariance matrix. If not provided the diagonal 
        identity matrix is used.

    path : array of floats (default=None)
        In "path" mode, an array of float values for scaling L.

    verbose : int (default=0)
        Verbosity level.

    method : one of 'quic'... (default=quic)

    metric : one of 'log_likelihood' (default), 'frobenius', 'spectral', 'kl', 
             or 'quadratic'
        Used in self.score().

    initialize_method : one of 'corrcoef', 'cov'

    Attributes
    ----------
    covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated covariance matrix
        If mode='path', this is 2D ndarray, shape (len(path), n_features ** 2)

    precision_ : 2D ndarray, shape (n_features, n_features)
        Estimated pseudo-inverse matrix.
        If mode='path', this is 2D ndarray, shape (len(path), n_features ** 2)

    opt_ :

    cputime_ :

    iters_ :    

    duality_gap_ :

    """
    def __init__(self, lam=0.5, mode='default', tol=1e-6, max_iter=1000,
                 Theta0=None, Sigma0=None, path=None, verbose=0, method='quic',
                 metric='log_likelihood', initialize_method='corrcoef'):
        self.lam = lam
        self.mode = mode
        self.tol = tol
        self.max_iter = max_iter
        self.Theta0 = Theta0
        self.Sigma0 = Sigma0
        self.path = path
        self.verbose = verbose
        self.method = method
        self.metric = metric
        self.initialize_method = initialize_method

        self.covariance_ = None
        self.precision_ = None
        self.opt_ = None
        self.cputime_ = None
        self.iters_ = None
        self.duality_gap_ = None
        self.score_best_path_scale_ = None
        self.score_best_path_scale_index_ = None

        # these must be updated upon self.fit()
        self.sample_covariance_ = None
        self.n_samples = None
        self.n_features = None
        self.is_fitted = False

        super(InverseCovariance, self).__init__()


    def initialize_coefficients(self, X):
        if self.initialize_method is 'corrcoef':
            return np.corrcoef(X), 1.0
        elif self.initialize_method is 'cov':   
            init_cov = np.cov(X, rowvar=False)
            return init_cov, np.max(np.triu(init_cov))
        else:
            raise ValueError("initialize_method must be 'corrcoeff' or 'cov'.")


    def fit(self, X, y=None, **fit_params):
        """Fits the inverse covariance model according to the given training 
        data and parameters.

        Parameters
        -----------
        X : 2D ndarray, shape (n_features, n_features)
            Input data.

        Returns
        -------
        self
        """
        X = check_array(X)
        X = as_float_array(X, copy=False, force_all_finite=False)
        self.n_samples, self.n_features = np.shape(X)
        self.sample_covariance_, scale_lam = self.initialize_coefficients(X)
        if self.method is 'quic':
            (self.precision_, self.covariance_, self.opt_, self.cputime_, 
            self.iters_, self.duality_gap_) = quic(self.sample_covariance_,
                                                self.lam * scale_lam,
                                                mode=self.mode,
                                                tol=self.tol,
                                                max_iter=self.max_iter,
                                                Theta0=self.Theta0,
                                                Sigma0=self.Sigma0,
                                                path=self.path,
                                                msg=self.verbose)
        else:
            raise NotImplementedError(
                "Only method='quic' has been implemented.")

        self.is_fitted = True
        return self


    def score(self, X_test, y=None):
        """Computes the score between cov/prec of sample covariance of X_test 
        and X via 'metric'.

        Note: We want to maximize score so we return the negative error 
              (or the max negative error). 
       
        ----------
        X_test : array-like, shape = [n_samples, n_features]
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).
        
        y : not used.
        
        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.
        """        
        S_test, scale_lam = self.initialize_coefficients(X_test)
        error = self.error_norm(
                S_test,
                norm=self.metric,
                scaling=False, 
                squared=False)

        if self.mode is not 'path':
            return -error

        path_errors = [-e for e in error]
        min_lidx = np.argmin(path_errors)
        self.score_best_path_scale_index_ = min_lidx
        self.score_best_path_scale_ = self.path[min_lidx]
        return path_errors[min_lidx]


    def error_norm(self, comp_cov, norm='frobenius', scaling=False, 
                   squared=True):
        """Computes the error between two inverse-covariance estimators 
        (i.e., via covariance).
        
        Parameters
        ----------
        comp_cov : array-like, shape = (n_features, n_features)
            The precision to compare with.
            This should normally be the test sample covariance/precision.
                
        scaling : bool
            If True, the squared error norm is divided by n_features.
            If False (default), the squared error norm is not rescaled.

        norm : str
            The type of norm used to compute the error between the estimated 
            self.precision, self.covariance and the reference `comp_cov`. 
            Available error types:
            
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            - 'kl': kl-divergence 
            - 'quadratic': quadratic loss
            - 'log_likelihood': negative log likelihood

            The term 'norm' is retained to be compatible with EmpiricalCovariance
            but 'metric' would be more appropriate.
        
        squared : bool
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.
        
        Returns
        -------
        The min error between `self.covariance_` and `comp_cov` 
        If mode='path', this will also set self.score_best_path_scale_ with the 
        best lambda for the current min error.
        """
        def _compute_error(comp_cov, self_prec=None, self_cov=None):
            if norm == "frobenius":
                error = comp_cov - self_cov
                result = np.sum(error ** 2)
                
                if not squared:
                    result = np.sqrt(result)

                if scaling:
                    result = result / error.shape[0]

                return result
            
            elif norm == "spectral":
                error = comp_cov - self_cov
                result = np.amax(np.linalg.svdvals(np.dot(error.T, error)))
                
                if not squared:
                    result = np.sqrt(result)

                if scaling:
                    result = result / error.shape[0]

                return result
            
            elif norm == "kl":
                return kl_loss(comp_cov, self_prec)
            elif norm == "quadratic":
                return quadratic_loss(comp_cov, self_prec)
            elif norm == "log_likelihood":
                return -log_likelihood(comp_cov, self_prec)
            else:
                raise NotImplementedError(
                    "Must be frobenius, spectral, kl, quadratic, or\
                    log_likelihood")
  

        if self.mode is not 'path':
            return _compute_error(comp_cov,
                                self_prec=self.precision_,
                                self_cov=self.covariance_)

        path_errors = []
        for lidx, lam_scale in enumerate(self.path):
            dim, _ = comp_cov.shape
            self_prec = np.reshape(self.precision_[lidx, :], (dim, dim))
            self_cov = np.reshape(self.covariance_[lidx, :], (dim, dim))
            path_errors.append(_compute_error(comp_cov,
                                            self_prec=self_prec,
                                            self_cov=self_cov))

        return path_errors

    def model_select(self, gamma=0):
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
        Lambda and lambda-index that minimizes ebic score (float).
        '''
        # must be path mode
        if self.mode is not 'path':
            raise NotImplementedError("Model selection only implemented for\
                                      mode=path")
            return

        # model must be fitted
        if not self.is_fitted:
            return

        ebic_scores = []
        for lidx, lam_scale in enumerate(self.path):
            dim, _ = self.sample_covariance_.shape
            prec = np.reshape(self.precision_[lidx, :], (dim, dim))
            #cov = np.reshape(self.covariance_[lidx, :], (dim, dim))
            ebic_scores.append(ebic(
                    self.sample_covariance_,
                    prec,
                    self.n_samples,
                    self.n_features,
                    gamma=gamma))

        min_lidx = np.argmin(ebic_scores)
        return self.path[min_lidx], min_lidx
