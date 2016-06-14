import numpy as np
from sklearn.base import BaseEstimator 
from sklearn.utils import check_array, as_float_array
from sklearn.utils.extmath import fast_logdet

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
    
    # NOTE TO MANJARI: 
    # - scikit learn version does some additional scaling and normalization
    #   is this something we need to do?
    # - should this just be the same one used in Empirical Covariance?

    assert covariance.shape == precision.shape
    return np.trace(covariance * precision) - fast_logdet(precision)


def kl_loss(precision_estimate, precision):
    """Computes ...
    
    # Trace(That^{-1}T) - log (That^{-1}T) - p

    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance
    
    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested
    
    Returns
    -------
    KL-divergence between precision_estimate and precision
    """
    assert precision_estimate.shape == precision.shape
    dim, _ = precision.shape
    # T \ T_hat = T_hat^{-1} * T
    ThinvT = np.linalg.solve(precision, precision_estimate) 
    return np.trace(ThinvT) - log(ThinvT) - dim


def quadratic_loss(covariance, precision):
    """Computes ...
    
    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance
    
    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested
    
    Returns
    -------
    Quadratic loss
    """
    assert covariance.shape == precision.shape
    dim, _ = precision.shape
    return np.trace((covariance * precision - np.eye(dim))**2)


def quic(S, L, mode='default', tol=1e-6, max_iter=1000, X0=None, W0=None,\
        path=None, msg=0):
    
    assert mode in ['default', 'path', 'trace'],\
            'mode = \'default\', \'path\' or \'trace\'.'

    Sn, Sm = S.shape
    assert Sn == Sm, 'Expected a square empircal covariance matrix S.'

    # Regularization parameter matrix L.
    if isinstance(L, float):
        _L = np.empty((Sn, Sm))
        _L[:] = L
    else:
        assert L.shape == S.shape, 'L, S shape mismatch.'
        _L = as_float_array(L, copy=False, force_all_finite=False)
 
    # Defaults.
    optSize = 1
    iterSize = 1
    if mode is "trace":
        optSize = max_iter

    # Default X0, W0 when both are None.
    if X0 is None and W0 is None:
        X0 = np.eye(Sn)
        W0 = np.eye(Sn)

    assert X0 is not None, 'X0 and W0 must both be None or both specified.'
    assert W0 is not None, 'X0 and W0 must both be None or both specified.'
    assert X0.shape == S.shape, 'X0, S shape mismatch.'
    assert W0.shape == S.shape, 'X0, S shape mismatch.'
    X0 = as_float_array(X0, copy=False, force_all_finite=False)
    W0 = as_float_array(W0, copy=False, force_all_finite=False)

    if mode is 'path':
        assert path is not None, 'Please specify the path scaling values.'
        path_len = len(path)
        optSize = path_len
        iterSize = path_len

        # Note here: memory layout is important:
        # a row of X/W holds a flattened Sn x Sn matrix,
        # one row for every element in _path_.
        X = np.empty((path_len, Sn * Sn))
        X[0,:] = X0.ravel()
        W = np.empty((path_len, Sn * Sn))
        W[0,:] = W0.ravel()
    else:
        path = np.empty(1)
        path_len = len(path)

        X = np.empty(X0.shape)
        X[:] = X0
        W = np.empty(W0.shape)
        W[:] = W0
                    
    # Run QUIC.
    opt = np.zeros(optSize)
    cputime = np.zeros(optSize)
    dGap = np.zeros(optSize)
    iters = np.zeros(iterSize, dtype=np.uint32)
    pyquic.quic(mode, Sn, S, _L, path_len, path, tol, msg, max_iter,
                X, W, opt, cputime, iters, dGap)

    if optSize == 1:
        opt = opt[0]
        cputime = cputime[0]
        dGap = dGap[0]

    if iterSize == 1:
        iters = iters[0]

    return X, W, opt, cputime, iters, dGap


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

    TODO: X0 = Theta0, W0 = S0
    X0 : 2D ndarray, shape (n_features, n_features) (default=None) 
        Initial guess for the inverse covariance matrix. If not provided, the 
        diagonal identity matrix is used.

    W0 : 2D ndarray, shape (n_features, n_features) (default=None)
        Initial guess for the covariance matrix. If not provided the diagonal 
        identity matrix is used.

    path : array of floats (default=None)
        In "path" mode, an array of float values for scaling L.

    verbose : int (default=0)
        Verbosity level.

    method : one of 'quic', 'quicanddirty', 'ETC' (default=quic)

    Attributes
    ----------
    covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : 2D ndarray, shape (n_features, n_features)
        Estimated pseudo-inverse matrix.

    opt_ :

    cputime_ :

    iters_ :    

    duality_gap_ :

    """
    def __init__(self, lam=0.5, mode='default', tol=1e-6, max_iter=1000,
                 X0=None, W0=None, path=None, verbose=0, method='quic'):
        self.lam = lam
        self.mode = mode
        self.tol = tol
        self.max_iter = max_iter
        self.X0 = X0
        self.W0 = W0
        self.path = path
        self.verbose = verbose
        self.method = method

        self.covariance_ = None
        self.precision_ = None
        self.opt_ = None
        self.cputime_ = None
        self.iters_ = None
        self.duality_gap_ = None

        super(InverseCovariance, self).__init__()

    def fit(self, X, y=None, **fit_params):
        """Fits the inverse covariance model according to the given training 
        data and parameters.

        Parameters
        -----------
        X : 2D ndarray, shape (n_features, n_features)
            Input data.
        """
        X = check_array(X)
        X = as_float_array(X, copy=False, force_all_finite=False)

        n_samples, n_features = X.shape
        if n_samples != n_features:
            raise ValueError("Input data must be square. X shape = {}".format(
                             X.shape))
            return

        if self.method is 'quic':
            (self.precision_, self.covariance_, self.opt_, self.cputime_, 
            self.iters_, self.duality_gap_) = quic(X,
                                                self.lam,
                                                mode=self.mode,
                                                tol=self.tol,
                                                max_iter=self.max_iter,
                                                X0=self.X0,
                                                W0=self.W0,
                                                path=self.path,
                                                msg=self.verbose)
        else:
            raise NotImplementedError(
                "Only method='quic' has been implemented.")

        return self


    # ADDITIONAL METHODS WE COULD PROVIDE

    def score(self, X_test, y=None):
        """Computes the log-likelihood 

        # TODO: -log_likelihood instead?
       
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
        # TODO: As Manjari mentioned, we should take input data to the interface
        #       and spit out results.  This should make this make more sense.

        # compute empirical covariance of the test set
        #test_cov = empirical_covariance(
        #    X_test - self.location_, assume_centered=True)
        
        return log_likelihood(test_cov, self.precision_)


    def error_norm(self, comp_prec, norm='frobenius', scaling=True, 
                   squared=True):
        """Computes the error between two inverse covariance estimators 
        (i.e., over the precision).
        
        Parameters
        ----------
        comp_prec : array-like, shape = [n_features, n_features]
            The precision to compare with.
                
        scaling : bool
            If True (default), the squared error norm is divided by n_features.
            If False, the squared error norm is not rescaled.

        norm : str
            The type of norm used to compute the error. Available error types:
            - 'frobenius' (default): sqrt(tr(A^t.A))
            - 'spectral': sqrt(max(eigenvalues(A^t.A))
            - 'kl': kl-divergence 
            where A is the error ``(comp_prec - self.precision_)``.
        
        squared : bool
            Whether to compute the squared error norm or the error norm.
            If True (default), the squared error norm is returned.
            If False, the error norm is returned.
        
        Returns
        -------
        The error between `self.precision_` and `comp_prec` 
        """
        # compute the error
        error = comp_prec - self.precision_
        
        # compute the error norm
        if norm == "frobenius":
            result = np.sum(error ** 2)
        elif norm == "spectral":
            result = np.amax(linalg.svdvals(np.dot(error.T, error)))
        elif norm == "kl":
            result = kl_loss(self.precision_, comp_prec)
        else:
            raise NotImplementedError(
                "Only spectral and frobenius norms are implemented")

        # optionally scale the error norm
        if scaling:
            result = result / error.shape[0]
        
        # finally get either the squared norm or the norm
        if not squared:
            result = np.sqrt(squared_norm)

        return result

