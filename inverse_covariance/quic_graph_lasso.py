import time
import collections
import operator
import numpy as np

from sklearn.covariance import EmpiricalCovariance
from sklearn.utils import check_array, as_float_array
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.externals.joblib import Parallel, delayed
#from sklearn.model_selection import check_cv, cross_val_score # >= 0.18
from sklearn.cross_validation import check_cv, cross_val_score # < 0.18

import pyquic
from inverse_covariance import (
    InverseCovarianceEstimator,
    _initialize_coefficients,
    _compute_error,
)


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
        assert lam.shape == S.shape, 'lam, S shape mismatch.'
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

        # path must be sorted from largest to smallest and have unique values
        check_path = sorted(set(path), reverse=True)
        assert_array_almost_equal(check_path, path)

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

    # reshape Theta, Sigma in path mode
    Theta_out = Theta 
    Sigma_out = Sigma
    if mode is 'path':
        Theta_out = []
        Sigma_out = []
        for lidx in range(path_len):
            Theta_out.append(np.reshape(Theta[lidx, :], (Sn, Sn)))
            Sigma_out.append(np.reshape(Sigma[lidx, :], (Sn, Sn)))

    return Theta_out, Sigma_out, opt, cputime, iters, dGap


class QuicGraphLasso(InverseCovarianceEstimator):
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

    method : one of 'quic'... (default=quic)

    verbose : integer
        Used in quic routine.

    score_metric : one of 'log_likelihood' (default), 'frobenius', 'spectral',
                  'kl', or 'quadratic'
        Used for computing self.score().

    initialize_method : one of 'corrcoef', 'cov'
        Computes initial covariance and scales lambda appropriately.


    Attributes
    ----------
    covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated covariance matrix
        If mode='path', this is 2D ndarray, shape (len(path), n_features ** 2)

    precision_ : 2D ndarray, shape (n_features, n_features)
        Estimated pseudo-inverse matrix.
        If mode='path', this is 2D ndarray, shape (len(path), n_features ** 2)

    sample_covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated sample covariance matrix

    lam_scale_ : (float)
        Additional scaling factor on lambda (due to magnitude of 
        sample_covariance_ values).

    path_ : None or array of floats
        Sorted (largest to smallest) path.  This will be None if not in path
        mode.

    opt_ :

    cputime_ :

    iters_ :    

    duality_gap_ :

    """
    def __init__(self, lam=0.5, mode='default', tol=1e-6, max_iter=1000,
                 Theta0=None, Sigma0=None, path=None, method='quic', verbose=0,
                 score_metric='log_likelihood', initialize_method='corrcoef'):
        # quic-specific params
        self.tol = tol
        self.max_iter = max_iter
        self.Theta0 = Theta0
        self.Sigma0 = Sigma0
        self.method = method
        self.verbose = verbose

        # quic-specific outputs
        self.opt_ = None
        self.cputime_ = None
        self.iters_ = None
        self.duality_gap_ = None

        # these must be updated upon self.fit()
        self.sample_covariance_ = None
        self.lam_scale_ = None
        self.n_samples = None
        self.n_features = None
        self.is_fitted = False

        super(QuicGraphLasso, self).__init__(lam=lam, mode=mode,
                score_metric=score_metric, path=path, 
                initialize_method=initialize_method)


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
        X = check_array(X, ensure_min_features=2, estimator=self)
        X = as_float_array(X, copy=False, force_all_finite=False)
        self.initialize_coefficients(X)
        if self.method is 'quic':
            (self.precision_, self.covariance_, self.opt_, self.cputime_, 
            self.iters_, self.duality_gap_) = quic(self.sample_covariance_,
                                                self.lam * self.lam_scale_,
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


def _quic_path(X, path, X_test=None, tol=1e-6,
         max_iter=1000, Theta0=None, Sigma0=None, method='quic', 
         verbose=0, score_metric='log_likelihood',
         initialize_method='corrcoef'):
    """Wrapper to compute path for example X.
    """
    S, lam_scale_ = _initialize_coefficients(
            X,
            method=initialize_method)

    path = path.copy(order='C')
    if method == 'quic':
        (precisions_, covariances_, opt_, cputime_, 
        iters_, duality_gap_) = quic(S,
                                    1.0,
                                    mode='path',
                                    tol=tol,
                                    max_iter=max_iter,
                                    Theta0=Theta0,
                                    Sigma0=Sigma0,
                                    path=path,
                                    msg=verbose)
    else:
        raise NotImplementedError(
            "Only method='quic' has been implemented.")

    if X_test is not None:
        S_test, lam_scale_test = _initialize_coefficients(
            X_test,
            method=initialize_method)
        
        path_errors = []
        for lidx, lam in enumerate(path):
            path_errors.append(_compute_error(S_test,
                                            covariances_[lidx],
                                            precisions_[lidx],
                                            score_metric=score_metric))
        scores_ = [-e for e in path_errors]

        return covariances_, precisions_, scores_
    
    return covariances_, precisions_

class QuicGraphLassoCV(InverseCovarianceEstimator):
    """Sparse inverse covariance w/ cross-validated choice of the l1 penalty
    via quadratic approximation.  

    This takes advantage of "path" mode in QuicGraphLasso.
    See sklearn.covariance.graph_lasso.GraphLassoCV.

    Parameters
    -----------  
    lams : integer, or list positive float, optional
        If an integer is given, it fixes the number of points on the
        grids of alpha to be used. If a list is given, it gives the
        grid to be used. See the notes in the class docstring for
        more details.
    
    n_refinements: strictly positive integer
        The number of times the grid is refined. Not used if explicit
        values of alphas are passed.

    n_jobs: int, optional
        number of jobs to run in parallel (default 1).

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

    method : one of 'quic'... (default=quic)

    score_metric : one of 'log_likelihood' (default), 'frobenius', 'spectral',
                  'kl', or 'quadratic'
        Used for computing self.score().

    initialize_method : one of 'corrcoef', 'cov'
        Computes initial covariance and scales lambda appropriately.

    Attributes
    ----------    
    covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : 2D ndarray, shape (n_features, n_features)
        Estimated pseudo-inverse matrix.

    lam_ : float
        Penalization parameter selected.
    
    cv_lams_ : list of float
        All penalization parameters explored.
    
    `grid_scores`: 2D numpy.ndarray (n_alphas, n_folds)
        Log-likelihood score on left-out data across folds.
    
    n_iter_ : int
        Number of iterations run for the optimal alpha.
    """
    def __init__(self, lams=4, n_refinements=4, cv=None, tol=1e-6,
                 max_iter=1000, Theta0=None, Sigma0=None, method='quic', 
                 verbose=0, n_jobs=1, score_metric='log_likelihood',
                 initialize_method='corrcoef'):
        # GridCV params
        self.n_jobs = n_jobs
        self.cv = cv
        self.lams = lams
        self.n_refinements = n_refinements

        # quic-specific params
        self.tol = tol
        self.max_iter = max_iter
        self.Theta0 = Theta0
        self.Sigma0 = Sigma0
        self.method = method
        self.verbose = verbose

        # quic-specific outputs
        self.opt_ = None
        self.cputime_ = None
        self.iters_ = None
        self.duality_gap_ = None

        # these must be updated upon self.fit()
        self.sample_covariance_ = None
        self.lam_scale_ = None
        self.n_samples = None
        self.n_features = None
        self.is_fitted = False

        super(QuicGraphLassoCV, self).__init__(lam=1.0, mode='path',
                score_metric=score_metric, path=[1.0], 
                initialize_method=initialize_method)


    def fit(self, X, y=None):
        """Fits the GraphLasso covariance model to X.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        """
        # initialize
        cv = check_cv(self.cv, X, y, classifier=False)
        X = check_array(X, ensure_min_features=2, estimator=self)
        X = as_float_array(X, copy=False, force_all_finite=False)
        self.initialize_coefficients(X)

        # get path
        if isinstance(self.lams, collections.Sequence):
            path = self.lams
            n_refinements = 1
        else:
            n_refinements = self.n_refinements
            lam_1 = np.max(np.abs(self.sample_covariance_.flat)) #self.lam_scale_
            lam_0 = 1e-2 * lam_1
            path = np.logspace(np.log10(lam_0), np.log10(lam_1), self.lams)[::-1]

        # run this thing a bunch
        results = list()
        t0 = time.time()
        for rr in range(n_refinements):
            # parallel version
            this_result = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )(
                delayed(_quic_path)(
                    X[train],
                    path,
                    X_test=X[test],
                    tol=self.tol, max_iter=self.max_iter, Theta0=self.Theta0,
                    Sigma0=self.Sigma0, method=self.method, verbose=self.verbose,
                    score_metric=self.score_metric,
                    initialize_method=self.initialize_method)
                for train, test in cv)

            # Little dance to transform the list in what we need
            covs, _, scores = zip(*this_result)
            covs = zip(*covs)
            scores = zip(*scores)
            results.extend(zip(path, scores, covs))
            results = sorted(results, key=operator.itemgetter(0), reverse=True)

            # Find the maximum (avoid using built in 'max' function to
            # have a fully-reproducible selection of the smallest alpha
            # in case of equality)
            best_score = -np.inf
            last_finite_idx = 0
            for index, (lam, scores, _) in enumerate(results):
                this_score = np.mean(scores)
                if this_score >= .1 / np.finfo(np.float64).eps:
                    this_score = np.nan
                
                if np.isfinite(this_score):
                    last_finite_idx = index
                
                if this_score >= best_score:
                    best_score = this_score
                    best_index = index

            # Refine the grid
            if best_index == 0:
                # We do not need to go back: we have chosen
                # the highest value of lambda for which there are
                # non-zero coefficients
                lam_1 = results[0][0]
                lam_0 = results[1][0]
            
            elif (best_index == last_finite_idx
                    and not best_index == len(results) - 1):
                # We have non-converged models on the upper bound of the
                # grid, we need to refine the grid there
                lam_1 = results[best_index][0]
                lam_0 = results[best_index + 1][0]
            
            elif best_index == len(results) - 1:
                lam_1 = results[best_index][0]
                lam_0 = 0.01 * results[best_index][0]
            
            else:
                lam_1 = results[best_index - 1][0]
                lam_0 = results[best_index + 1][0]

            if not isinstance(self.lams, collections.Sequence):
                path = np.logspace(np.log10(lam_1), np.log10(lam_0),
                                     self.lams + 2)
                path = path[1:-1]

            if self.verbose and n_refinements > 1:
                print('[GraphLassoCV] Done refinement % 2i out of %i: % 3is'
                      % (rr + 1, n_refinements, time.time() - t0))

        results = list(zip(*results))
        grid_scores = list(results[1])
        lams = list(results[0])
        
        # Finally, compute the score with lambda = 0
        lams.append(0)
        grid_scores.append(cross_val_score(EmpiricalCovariance(), X,
                                           cv=cv, n_jobs=self.n_jobs))
        self.grid_scores = np.array(grid_scores)
        self.lam_ = lams[best_index]
        self.cv_lams_ = lams

        # Finally fit the model with the selected lambda
        if self.method is 'quic':
            (self.precision_, self.covariance_, self.opt_, self.cputime_, 
            self.iters_, self.duality_gap_) = quic(self.sample_covariance_,
                                                self.lam_,
                                                mode='default',
                                                tol=self.tol,
                                                max_iter=self.max_iter,
                                                Theta0=self.Theta0,
                                                Sigma0=self.Sigma0,
                                                path=None,
                                                msg=self.verbose)
        else:
            raise NotImplementedError(
                "Only method='quic' has been implemented.")

        self.is_fitted = True
        return self
