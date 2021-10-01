from __future__ import absolute_import

import sys
import time
import operator
import numpy as np
from functools import partial

from sklearn.covariance import EmpiricalCovariance
from sklearn.utils import check_array, as_float_array, deprecated
from numpy.testing import assert_array_almost_equal
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, RepeatedKFold

from . import pyquic
from .inverse_covariance import (
    InverseCovarianceEstimator,
    _init_coefs,
    _compute_error,
    _validate_path,
)


def quic(
    S,
    lam,
    mode="default",
    tol=1e-6,
    max_iter=1000,
    Theta0=None,
    Sigma0=None,
    path=None,
    msg=0,
):
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
    assert mode in ["default", "path", "trace"], "mode = 'default', 'path' or 'trace'."

    Sn, Sm = S.shape
    if Sn != Sm:
        raise ValueError("Input data must be square. S shape = {}".format(S.shape))
        return

    # Regularization parameter matrix L.
    if isinstance(lam, float):
        _lam = np.empty((Sn, Sm))
        _lam[:] = lam
        _lam[np.diag_indices(Sn)] = 0.  # make sure diagonal is zero
    else:
        assert lam.shape == S.shape, "lam, S shape mismatch."
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

    assert Theta0 is not None, "Theta0 and Sigma0 must both be None or both specified."
    assert Sigma0 is not None, "Theta0 and Sigma0 must both be None or both specified."
    assert Theta0.shape == S.shape, "Theta0, S shape mismatch."
    assert Sigma0.shape == S.shape, "Theta0, Sigma0 shape mismatch."
    Theta0 = as_float_array(Theta0, copy=False, force_all_finite=False)
    Sigma0 = as_float_array(Sigma0, copy=False, force_all_finite=False)

    if mode == "path":
        assert path is not None, "Please specify the path scaling values."

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
        Theta[0, :] = Theta0.ravel()
        Sigma = np.empty((path_len, Sn * Sn))
        Sigma[0, :] = Sigma0.ravel()
    else:
        path = np.empty(1)
        path_len = len(path)

        Theta = np.empty(Theta0.shape)
        Theta[:] = Theta0
        Sigma = np.empty(Sigma0.shape)
        Sigma[:] = Sigma0

    # Cython fix for Python3
    # http://cython.readthedocs.io/en/latest/src/tutorial/strings.html
    quic_mode = mode
    if sys.version_info[0] >= 3:
        quic_mode = quic_mode.encode("utf-8")

    # Run QUIC.
    opt = np.zeros(optSize)
    cputime = np.zeros(optSize)
    dGap = np.zeros(optSize)
    iters = np.zeros(iterSize, dtype=np.uint32)
    pyquic.quic(
        quic_mode,
        Sn,
        S,
        _lam,
        path_len,
        path,
        tol,
        msg,
        max_iter,
        Theta,
        Sigma,
        opt,
        cputime,
        iters,
        dGap,
    )

    if optSize == 1:
        opt = opt[0]
        cputime = cputime[0]
        dGap = dGap[0]

    if iterSize == 1:
        iters = iters[0]

    # reshape Theta, Sigma in path mode
    Theta_out = Theta
    Sigma_out = Sigma
    if mode == "path":
        Theta_out = []
        Sigma_out = []
        for lidx in range(path_len):
            Theta_out.append(np.reshape(Theta[lidx, :], (Sn, Sn)))
            Sigma_out.append(np.reshape(Sigma[lidx, :], (Sn, Sn)))

    return Theta_out, Sigma_out, opt, cputime, iters, dGap


class QuicGraphicalLasso(InverseCovarianceEstimator):
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

        If a scalar lambda is used, a penalty matrix will be generated
        containing lambda for all values in both upper and lower triangles
        and zeros along the diagonal.  This differs from the scalar graphical
        lasso by the diagonal. To replicate the scalar formulation you must
        manualy pass in lam * np.ones((n_features, n_features)).

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
        In "path" mode, an array of float values for scaling lam.
        The path must be sorted largest to smallest.  This class will auto sort
        this, in which case indices correspond to self.path_

    method : 'quic' or 'bigquic', ... (default=quic)
        Currently only 'quic' is supported.

    verbose : integer
        Used in quic routine.

    score_metric : one of 'log_likelihood' (default), 'frobenius', 'spectral',
                  'kl', or 'quadratic'
        Used for computing self.score().

    init_method : one of 'corrcoef', 'cov', 'spearman', 'kendalltau',
        or a custom function.
        Computes initial covariance and scales lambda appropriately.
        Using the custom function extends graphical model estimation to
        distributions beyond the multivariate Gaussian.
        The `spearman` or `kendalltau` options extend inverse covariance
        estimation to nonparanormal and transelliptic graphical models.
        Custom function must return ((n_features, n_features) ndarray, float)
        where the scalar parameter will be used to scale the penalty lam.

    auto_scale : bool
        If True, will compute self.lam_scale_ = max off-diagonal value when
        init_method='cov'.
        If false, then self.lam_scale_ = 1.
        lam_scale_ is used to scale user-supplied self.lam during fit.

    Methods
    ----------
    lam_at_index(lidx) :  Compute the scaled lambda used at index lidx.
        The parameter lidx is ignored when mode='default'.  Can use self.lam_
        for convenience in this case.

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

    lam_ : (float) or 2D ndarray, shape (n_features, n_features)
        When mode='default', this is the lambda used in fit (lam * lam_scale_)

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

    def __init__(
        self,
        lam=0.5,
        mode="default",
        tol=1e-6,
        max_iter=1000,
        Theta0=None,
        Sigma0=None,
        path=None,
        method="quic",
        verbose=0,
        score_metric="log_likelihood",
        init_method="corrcoef",
        auto_scale=True,
    ):
        # quic-specific params
        self.lam = lam
        self.mode = mode
        self.tol = tol
        self.max_iter = max_iter
        self.Theta0 = Theta0
        self.Sigma0 = Sigma0
        self.method = method
        self.verbose = verbose
        self.path = path

        if self.mode == "path" and path is None:
            raise ValueError("path required in path mode.")
            return

        super(QuicGraphicalLasso, self).__init__(
            score_metric=score_metric, init_method=init_method, auto_scale=auto_scale
        )

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
        # quic-specific outputs
        self.opt_ = None
        self.cputime_ = None
        self.iters_ = None
        self.duality_gap_ = None

        # these must be updated upon self.fit()
        self.sample_covariance_ = None
        self.lam_scale_ = None
        self.is_fitted_ = False

        self.path_ = _validate_path(self.path)
        X = check_array(X, ensure_min_features=2, estimator=self)
        X = as_float_array(X, copy=False, force_all_finite=False)
        self.init_coefs(X)
        if self.method == "quic":
            (
                self.precision_,
                self.covariance_,
                self.opt_,
                self.cputime_,
                self.iters_,
                self.duality_gap_,
            ) = quic(
                self.sample_covariance_,
                self.lam * self.lam_scale_,
                mode=self.mode,
                tol=self.tol,
                max_iter=self.max_iter,
                Theta0=self.Theta0,
                Sigma0=self.Sigma0,
                path=self.path_,
                msg=self.verbose,
            )
        else:
            raise NotImplementedError("Only method='quic' has been implemented.")

        self.is_fitted_ = True
        return self

    def lam_at_index(self, lidx):
        """Compute the scaled lambda used at index lidx.
        """
        if self.path_ is None:
            return self.lam * self.lam_scale_

        return self.lam * self.lam_scale_ * self.path_[lidx]

    @property
    def lam_(self):
        if self.path_ is not None:
            print("lam_ is an invalid parameter in path mode, " "use self.lam_at_index")
        return self.lam_at_index(0)


@deprecated(
    "The class QuicGraphLasso is deprecated " "Use class QuicGraphicalLasso instead."
)
class QuicGraphLasso(QuicGraphicalLasso):
    pass


def _quic_path(
    X,
    path,
    X_test=None,
    lam=0.5,
    tol=1e-6,
    max_iter=1000,
    Theta0=None,
    Sigma0=None,
    method="quic",
    verbose=0,
    score_metric="log_likelihood",
    init_method="corrcoef",
):
    """Wrapper to compute path for example X.
    """
    S, lam_scale_ = _init_coefs(X, method=init_method)

    path = path.copy(order="C")

    if method == "quic":
        (precisions_, covariances_, opt_, cputime_, iters_, duality_gap_) = quic(
            S,
            lam,
            mode="path",
            tol=tol,
            max_iter=max_iter,
            Theta0=Theta0,
            Sigma0=Sigma0,
            path=path,
            msg=verbose,
        )
    else:
        raise NotImplementedError("Only method='quic' has been implemented.")

    if X_test is not None:
        S_test, lam_scale_test = _init_coefs(X_test, method=init_method)

        path_errors = []
        for lidx, lam in enumerate(path):
            path_errors.append(
                _compute_error(
                    S_test,
                    covariances_[lidx],
                    precisions_[lidx],
                    score_metric=score_metric,
                )
            )
        scores_ = [-e for e in path_errors]

        return covariances_, precisions_, scores_

    return covariances_, precisions_


def _quic_path_spark(indexed_params, quic_path, X_bc):
    index, (local_train, local_test) = indexed_params
    result = quic_path(X_bc.value[local_train], X_test=X_bc.value[local_test])
    return index, result


class QuicGraphicalLassoCV(InverseCovarianceEstimator):
    """Sparse inverse covariance w/ cross-validated choice of the l1 penalty
    via quadratic approximation.

    This takes advantage of "path" mode in QuicGraphicalLasso.
    See sklearn.covariance.graph_lasso.GraphLassoCV.

    Parameters
    -----------
    lam : scalar or 2D ndarray, shape (n_features, n_features) (default=0.5)
        Regularization parameters per element of the inverse covariance matrix.
        The next parameter 'lams' scale this matrix as the lasso path learned.

        If a scalar lambda is used, a penalty matrix will be generated
        containing lambda for all values in both upper and lower triangles
        and zeros along the diagonal.  This differs from the scalar graphical
        lasso by the diagonal. To replicate the scalar formulation you must
        manualy pass in lam * np.ones((n_features, n_features)).

    lams : integer, or list positive float, optional
        If an integer is given, it fixes the number of points on the
        grids of alpha to be used. If a list is given, it gives the
        grid to be used. See the notes in the class docstring for
        more details.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, use the default 3-fold, 3-trial RepeatedKFold cross-validation,
        - integer, to specify the number of folds (3-trial RepeatedKFold).
        - tuple, (n_folds, n_trials)
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

    n_refinements : strictly positive integer
        The number of times the grid is refined. Not used if explicit
        values of alphas are passed.

    n_jobs : int, optional
        number of jobs to run in parallel (default 1).

    sc : sparkContext (default=None)
        If not None and a valid SparkContext, the cross-validation iterations
        will be performed in parallel via Apache Spark.  In this case n_jobs
        will be unused.

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

    method : 'quic' or 'bigquic', ... (default=quic)
        Currently only 'quic' is supported.

    score_metric : one of 'log_likelihood' (default), 'frobenius', 'spectral',
                  'kl', or 'quadratic'
        Used for computing self.score().

    init_method : one of 'corrcoef', 'cov', 'spearman', 'kendalltau',
        or a custom function.
        Computes initial covariance and scales lambda appropriately.
        Using the custom function extends graphical model estimation to
        distributions beyond the multivariate Gaussian.
        The `spearman` or `kendalltau` options extend inverse covariance
        estimation to nonparanormal and transelliptic graphical models.
        Custom function must return ((n_features, n_features) ndarray, float)
        where the scalar parameter will be used to scale the penalty lam.

    auto_scale : bool
        If True, will compute self.lam_scale_ = max off-diagonal value when
        init_method='cov'.
        If false, then self.lam_scale_ = 1.
        lam_scale_ is used to scale user-supplied self.lam during fit.

    backend : string, optional (default=threading)
        Joblib parallelization backend.
        Not used when using the sparkContext (sc).

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

    grid_scores_: 2D numpy.ndarray (n_alphas, n_folds)
        Log-likelihood score on left-out data across folds.

    n_iter_ : int
        Number of iterations run for the optimal alpha.
    """

    def __init__(
        self,
        lam=1.0,
        lams=4,
        n_refinements=4,
        cv=None,
        tol=1e-6,
        max_iter=1000,
        Theta0=None,
        Sigma0=None,
        method="quic",
        verbose=0,
        n_jobs=1,
        sc=None,
        score_metric="log_likelihood",
        init_method="corrcoef",
        auto_scale=True,
        backend="threading",
    ):
        # GridCV params
        self.n_jobs = n_jobs
        self.sc = sc
        self.cv = cv
        self.lam = lam
        self.lams = lams
        self.n_refinements = n_refinements

        # quic-specific params
        self.tol = tol
        self.max_iter = max_iter
        self.Theta0 = Theta0
        self.Sigma0 = Sigma0
        self.method = method
        self.verbose = verbose
        self.backend = backend

        super(QuicGraphicalLassoCV, self).__init__(
            score_metric=score_metric, init_method=init_method, auto_scale=auto_scale
        )

    def fit(self, X, y=None):
        """Fits the GraphLasso covariance model to X.

        Closely follows sklearn.covariance.graph_lasso.GraphLassoCV.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        """
        # quic-specific outputs
        self.opt_ = None
        self.cputime_ = None
        self.iters_ = None
        self.duality_gap_ = None

        # these must be updated upon self.fit()
        self.sample_covariance_ = None
        self.lam_scale_ = None
        self.is_fitted_ = False

        # initialize
        X = check_array(X, ensure_min_features=2, estimator=self)
        X = as_float_array(X, copy=False, force_all_finite=False)

        if self.cv is None:
            cv = (3, 10)
        elif isinstance(self.cv, int):
            cv = (self.cv, 10)  # upgrade with default number of trials
        elif isinstance(self.cv, tuple):
            cv = self.cv

        cv = RepeatedKFold(n_splits=cv[0], n_repeats=cv[1])

        self.init_coefs(X)

        # get path
        if isinstance(self.lams, int):
            n_refinements = self.n_refinements
            lam_1 = self.lam_scale_
            lam_0 = 1e-2 * lam_1
            path = np.logspace(np.log10(lam_0), np.log10(lam_1), self.lams)[::-1]
        else:
            path = self.lams
            n_refinements = 1

        # run this thing a bunch
        results = list()
        t0 = time.time()
        for rr in range(n_refinements):
            if self.sc is None:
                # parallel version
                this_result = Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose, backend=self.backend
                )(
                    delayed(_quic_path)(
                        X[train],
                        path,
                        X_test=X[test],
                        lam=self.lam,
                        tol=self.tol,
                        max_iter=self.max_iter,
                        Theta0=self.Theta0,
                        Sigma0=self.Sigma0,
                        method=self.method,
                        verbose=self.verbose,
                        score_metric=self.score_metric,
                        init_method=self.init_method,
                    )
                    for train, test in cv.split(X)
                )
            else:
                # parallel via spark
                train_test_grid = [(train, test) for (train, test) in cv.split(X)]
                indexed_param_grid = list(
                    zip(range(len(train_test_grid)), train_test_grid)
                )
                par_param_grid = self.sc.parallelize(indexed_param_grid)
                X_bc = self.sc.broadcast(X)

                # wrap function parameters so we dont pick whole self object
                quic_path = partial(
                    _quic_path,
                    path=path,
                    lam=self.lam,
                    tol=self.tol,
                    max_iter=self.max_iter,
                    Theta0=self.Theta0,
                    Sigma0=self.Sigma0,
                    method=self.method,
                    verbose=self.verbose,
                    score_metric=self.score_metric,
                    init_method=self.init_method,
                )

                indexed_results = dict(
                    par_param_grid.map(
                        partial(_quic_path_spark, quic_path=quic_path, X_bc=X_bc)
                    ).collect()
                )
                this_result = [
                    indexed_results[idx] for idx in range(len(train_test_grid))
                ]
                X_bc.unpersist()

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
            best_index = 0
            for index, (lam, scores, _) in enumerate(results):
                # sometimes we get -np.inf in the result (in kl-loss)
                scores = [s for s in scores if not np.isinf(s)]
                if len(scores) == 0:
                    this_score = -np.inf
                else:
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

            elif best_index == last_finite_idx and not best_index == len(results) - 1:
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

            if isinstance(self.lams, int):
                path = np.logspace(np.log10(lam_1), np.log10(lam_0), self.lams + 2)
                path = path[1:-1]

            if self.verbose and n_refinements > 1:
                print(
                    "[GraphLassoCV] Done refinement % 2i out of %i: % 3is"
                    % (rr + 1, n_refinements, time.time() - t0)
                )

        results = list(zip(*results))
        grid_scores_ = list(results[1])
        lams = list(results[0])

        # Finally, compute the score with lambda = 0
        lams.append(0)
        grid_scores_.append(
            cross_val_score(EmpiricalCovariance(), X, cv=cv, n_jobs=self.n_jobs)
        )
        self.grid_scores_ = np.array(grid_scores_)
        self.lam_ = self.lam * lams[best_index]
        self.cv_lams_ = [self.lam * l for l in lams]

        # Finally fit the model with the selected lambda
        if self.method == "quic":
            (
                self.precision_,
                self.covariance_,
                self.opt_,
                self.cputime_,
                self.iters_,
                self.duality_gap_,
            ) = quic(
                self.sample_covariance_,
                self.lam_,
                mode="default",
                tol=self.tol,
                max_iter=self.max_iter,
                Theta0=self.Theta0,
                Sigma0=self.Sigma0,
                path=None,
                msg=self.verbose,
            )
        else:
            raise NotImplementedError("Only method='quic' has been implemented.")

        self.is_fitted_ = True
        return self


@deprecated(
    "The class QuicGraphLassoCV is deprecated "
    "Use class QuicGraphicalLassoCV instead."
)
class QuicGraphLassoCV(QuicGraphicalLassoCV):
    pass


class QuicGraphicalLassoEBIC(InverseCovarianceEstimator):
    """
    Computes a sparse inverse covariance matrix estimation using quadratic
    approximation and EBIC model selection. (Convenience Class)

    Note: This estimate can be obtained using the more general QuicGraphicalLasso
          estimator and taking advantage of `ebic_select()` and
          `lambda_at_index()` methods.

    See analogous sklearn.linear_model.LassoLarsIC.

    Parameters
    -----------
    lam : scalar or 2D ndarray, shape (n_features, n_features) (default=0.5)
        Regularization parameters per element of the inverse covariance matrix.

        If a scalar lambda is used, a penalty matrix will be generated
        containing lambda for all values in both upper and lower triangles
        and zeros along the diagonal.  This differs from the scalar graphical
        lasso by the diagonal. To replicate the scalar formulation you must
        manualy pass in lam * np.ones((n_features, n_features)).

    path : array of floats or int (default=100)
        An array of float values for scaling lam.
        An int will choose the number of log-scale points to fit.

    gamma : float (default=0)
        Extended Bayesian Information Criteria (EBIC) for model selection.
        Choice of gamma=0 leads to classical BIC
        Positive gamma leads to stronger penalization of large graphs.

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

    method : one of 'quic'... (default=quic)

    verbose : integer
        Used in quic routine.

    init_method : one of 'corrcoef', 'cov', 'spearman', 'kendalltau',
        or a custom function.
        Computes initial covariance and scales lambda appropriately.
        Using the custom function extends graphical model estimation to
        distributions beyond the multivariate Gaussian.
        The `spearman` or `kendalltau` options extend inverse covariance
        estimation to nonparanormal and transelliptic graphical models.
        Custom function must return ((n_features, n_features) ndarray, float)
        where the scalar parameter will be used to scale the penalty lam.

    auto_scale : bool
        If True, will compute self.lam_scale_ = max off-diagonal value when
        init_method='cov'.
        If false, then self.lam_scale_ = 1.
        lam_scale_ is used to scale user-supplied self.lam during fit.

    Attributes
    ----------
    covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : 2D ndarray, shape (n_features, n_features)
        Estimated pseudo-inverse matrix.

    sample_covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated sample covariance matrix

    lam_ : (float)
        Lambda chosen by EBIC (with scaling already applied).
    """

    def __init__(
        self,
        lam=1.0,
        path=100,
        gamma=0,
        tol=1e-6,
        max_iter=1000,
        Theta0=None,
        Sigma0=None,
        method="quic",
        verbose=0,
        score_metric="log_likelihood",
        init_method="corrcoef",
        auto_scale=True,
    ):
        # quic-specific params
        self.lam = lam
        self.tol = tol
        self.max_iter = max_iter
        self.Theta0 = Theta0
        self.Sigma0 = Sigma0
        self.method = method
        self.verbose = verbose
        self.path = path
        self.gamma = gamma

        super(QuicGraphicalLassoEBIC, self).__init__(
            init_method=init_method, score_metric=score_metric, auto_scale=auto_scale
        )

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
        # quic-specific outputs
        self.opt_ = None
        self.cputime_ = None
        self.iters_ = None
        self.duality_gap_ = None

        # these must be updated upon self.fit()
        self.path_ = None
        self.sample_covariance_ = None
        self.lam_scale_ = None
        self.lam_ = None
        self.is_fitted_ = False

        X = check_array(X, ensure_min_features=2, estimator=self)
        X = as_float_array(X, copy=False, force_all_finite=False)
        self.init_coefs(X)

        # either use passed in path, or make our own path
        lam_1 = self.lam_scale_
        lam_0 = 1e-2 * lam_1
        if self.path is None:
            self.path_ = np.logspace(np.log10(lam_0), np.log10(lam_1), 100)[::-1]
        elif isinstance(self.path, int):
            self.path_ = np.logspace(np.log10(lam_0), np.log10(lam_1), self.path)[::-1]
        else:
            self.path_ = self.path

        self.path_ = _validate_path(self.path_)

        # fit along the path, temporarily populate
        # self.precision_, self.covariance_ with path values so we can use our
        # inherited selection function
        if self.method == "quic":
            (self.precision_, self.covariance_, _, _, _, _) = quic(
                self.sample_covariance_,
                self.lam * self.lam_scale_,
                mode="path",
                tol=self.tol,
                max_iter=self.max_iter,
                Theta0=self.Theta0,
                Sigma0=self.Sigma0,
                path=self.path_,
                msg=self.verbose,
            )
            self.is_fitted_ = True
        else:
            raise NotImplementedError("Only method='quic' has been implemented.")

        # apply EBIC criteria
        best_lam_idx = self.ebic_select(gamma=self.gamma)
        self.lam_ = self.lam * self.lam_scale_ * self.path_[best_lam_idx]
        self.precision_ = self.precision_[best_lam_idx]
        self.covariance_ = self.covariance_[best_lam_idx]

        self.is_fitted_ = True
        return self


@deprecated(
    "The class QuicGraphLassoEBIC is deprecated "
    "Use class QuicGraphicalLassoEBIC instead."
)
class QuicGraphLassoEBIC(QuicGraphicalLassoEBIC):
    pass
