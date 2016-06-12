from sklearn.base import BaseEstimator 
from sklearn.utils import check_array, as_float_array
import py_quic


class QUIC(BaseEstimator):
    """
    Computes a sparse inverse covariance matrix estimation using quadratic 
    approximation. 

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

        super(QUIC, self).__init__()

    def fit(self, X, y=None, **fit_params):
        """Fits the QUIC covariance model according to the given training 
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

        if self.method == 'quic':
            (self.precision_, self.covariance_, self.opt_, self.cputime_, 
            self.iters_, self.duality_gap_) = py_quic.quic(X,
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

    # def score(self, X_test, y=None):
    #   """Computes the log-likelihood of a Gaussian data set with
    #    `self.covariance_` as an estimator of its covariance matrix.

    # def error_norm(self, comp_cov, norm='frobenius', scaling=True, 
    #                squared=True):
    #    """Computes the Mean Squared Error between two covariance estimators.
    #    (In the sense of the Frobenius norm).