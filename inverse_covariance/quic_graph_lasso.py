import numpy as np
from sklearn.utils import check_array, as_float_array
from sklearn.utils.testing import assert_array_almost_equal

import pyquic
from inverse_covariance import InverseCovarianceEstimator



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
                 Theta0=None, Sigma0=None, path=None, method='quic',
                 score_metric='log_likelihood', initialize_method='corrcoef'):
        # quic-specific params
        self.tol = tol
        self.max_iter = max_iter
        self.Theta0 = Theta0
        self.Sigma0 = Sigma0
        self.method = method

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
        X = check_array(X)
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


class QuicGraphLassoCV(QuicGraphLasso):
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
                 score_metric='log_likelihood', initialize_method='corrcoef'):
        self.lams = lams
        self.n_refinements = n_refinements

        super(QuicGraphLasso, self).__init__(self, 
                lam=1.0, mode='path', tol=tol, max_iter=max_iter,
                Theta0=Theta0, Sigma0=Sigma0, path=[], method=method,
                score_metric=score_metric, initialize_method=initialize_method)

    def fit(self, X, y=None):
        """Fits the GraphLasso covariance model to X.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        """
        '''
        # Covariance does not make sense for a single feature
        X = check_array(X, ensure_min_features=2, estimator=self)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        emp_cov = empirical_covariance(
            X, assume_centered=self.assume_centered)

        cv = check_cv(self.cv, y, classifier=False)

        # List of (alpha, scores, covs)
        path = list()
        n_alphas = self.alphas
        inner_verbose = max(0, self.verbose - 1)

        if isinstance(n_alphas, collections.Sequence):
            alphas = self.alphas
            n_refinements = 1
        else:
            n_refinements = self.n_refinements
            alpha_1 = alpha_max(emp_cov)
            alpha_0 = 1e-2 * alpha_1
            alphas = np.logspace(np.log10(alpha_0), np.log10(alpha_1),
                                 n_alphas)[::-1]

        t0 = time.time()
        for i in range(n_refinements):
            with warnings.catch_warnings():
                # No need to see the convergence warnings on this grid:
                # they will always be points that will not converge
                # during the cross-validation
                warnings.simplefilter('ignore', ConvergenceWarning)
                # Compute the cross-validated loss on the current grid

                # NOTE: Warm-restarting graph_lasso_path has been tried, and
                # this did not allow to gain anything (same execution time with
                # or without).
                this_path = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )(delayed(graph_lasso_path)(X[train], alphas=alphas,
                                            X_test=X[test], mode=self.mode,
                                            tol=self.tol,
                                            enet_tol=self.enet_tol,
                                            max_iter=int(.1 * self.max_iter),
                                            verbose=inner_verbose)
                  for train, test in cv.split(X, y))

            # Little danse to transform the list in what we need
            covs, _, scores = zip(*this_path)
            covs = zip(*covs)
            scores = zip(*scores)
            path.extend(zip(alphas, scores, covs))
            path = sorted(path, key=operator.itemgetter(0), reverse=True)

            # Find the maximum (avoid using built in 'max' function to
            # have a fully-reproducible selection of the smallest alpha
            # in case of equality)
            best_score = -np.inf
            last_finite_idx = 0
            for index, (alpha, scores, _) in enumerate(path):
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
                # the highest value of alpha for which there are
                # non-zero coefficients
                alpha_1 = path[0][0]
                alpha_0 = path[1][0]
            elif (best_index == last_finite_idx
                    and not best_index == len(path) - 1):
                # We have non-converged models on the upper bound of the
                # grid, we need to refine the grid there
                alpha_1 = path[best_index][0]
                alpha_0 = path[best_index + 1][0]
            elif best_index == len(path) - 1:
                alpha_1 = path[best_index][0]
                alpha_0 = 0.01 * path[best_index][0]
            else:
                alpha_1 = path[best_index - 1][0]
                alpha_0 = path[best_index + 1][0]

            if not isinstance(n_alphas, collections.Sequence):
                alphas = np.logspace(np.log10(alpha_1), np.log10(alpha_0),
                                     n_alphas + 2)
                alphas = alphas[1:-1]

            if self.verbose and n_refinements > 1:
                print('[GraphLassoCV] Done refinement % 2i out of %i: % 3is'
                      % (i + 1, n_refinements, time.time() - t0))

        path = list(zip(*path))
        grid_scores = list(path[1])
        alphas = list(path[0])
        # Finally, compute the score with alpha = 0
        alphas.append(0)
        grid_scores.append(cross_val_score(EmpiricalCovariance(), X,
                                           cv=cv, n_jobs=self.n_jobs,
                                           verbose=inner_verbose))
        self.grid_scores = np.array(grid_scores)
        best_alpha = alphas[best_index]
        self.alpha_ = best_alpha
        self.cv_alphas_ = alphas

        # Finally fit the model with the selected alpha
        self.covariance_, self.precision_, self.n_iter_ = graph_lasso(
            emp_cov, alpha=best_alpha, mode=self.mode, tol=self.tol,
            enet_tol=self.enet_tol, max_iter=self.max_iter,
            verbose=inner_verbose, return_n_iter=True)
        return self
        '''
