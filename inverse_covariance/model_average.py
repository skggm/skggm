from __future__ import absolute_import

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_array, as_float_array
from sklearn.externals.joblib import Parallel, delayed
from functools import partial

from .inverse_covariance import _init_coefs
from . import QuicGraphLasso


def _check_psd(m):
    return np.all(np.linalg.eigvals(m) >= 0)


def _fully_random_weights(n_features, lam_scale, prng):
    """Generate a symmetric random matrix with zeros along the diagonal."""
    weights = np.zeros((n_features, n_features))
    n_off_diag = int((n_features ** 2 - n_features) / 2)
    weights[np.triu_indices(n_features, k=1)] =\
        0.1 * lam_scale * prng.randn(n_off_diag) + (0.25 * lam_scale)
    weights[weights < 0] = 0
    weights = weights + weights.T
    return weights


def _random_weights(n_features, lam, lam_perturb, prng):
    """Generate a symmetric random matrix with zeros along the diagnoal and
    non-zero elements take the value {lam * lam_perturb, lam / lam_perturb}
    with probability 1/2.
    """
    weights = np.zeros((n_features, n_features))
    n_off_diag = int((n_features ** 2 - n_features) / 2)
    berns = prng.binomial(1, 0.5, size=n_off_diag)
    vals = np.zeros(berns.shape)
    vals[berns == 0] = 1. * lam * lam_perturb
    vals[berns == 1] = 1. * lam / lam_perturb
    weights[np.triu_indices(n_features, k=1)] = vals
    weights[weights < 0] = 0
    weights = weights + weights.T
    return weights


def _fix_weights(weight_fun, *args):
    """Ensure random weight matrix is valid.

    TODO:  The diagonally dominant tuning currently doesn't make sense.
           Our weight matrix has zeros along the diagonal, so multiplying by
           a diagonal matrix results in a zero-matrix.
    """
    weights = weight_fun(*args)

    # TODO: fix this
    # disable checks for now
    return weights

    # if positive semidefinite, then we're good as is
    if _check_psd(weights):
        return weights

    # make diagonally dominant
    off_diag_sums = np.sum(weights, axis=1)  # NOTE: assumes diag is zero
    mod_mat = np.linalg.inv(np.sqrt(np.diag(off_diag_sums)))
    return np.dot(mod_mat, weights, mod_mat)


def _default_bootstrap(n_samples, num_subsamples, prng):
    """Returns an array of integers (0, n_samples-1)^num_subsamples."""
    return prng.permutation(n_samples)[:num_subsamples]


def _fit(indexed_params, penalization, lam, lam_perturb, lam_scale_, estimator,
         penalty_name, subsample, bootstrap, prng, X=None):
    """Wrapper function outside of instance for fitting a single model average
    trial.

    If X is None, then we assume we are using a broadcast spark object. Else,
    we expect X to get passed into this function.
    """
    index = indexed_params

    if isinstance(X, np.ndarray):
        local_X = X
    else:
        local_X = X.value

    n_samples, n_features = local_X.shape

    prec_is_real = False
    while not prec_is_real:
        boot_lam = None
        if penalization == 'subsampling':
            pass
        elif penalization == 'random':
            boot_lam = _fix_weights(_random_weights,
                                    n_features,
                                    lam,
                                    lam_perturb,
                                    prng)
        elif penalization == 'fully-random':
            boot_lam = _fix_weights(_fully_random_weights,
                                    n_features,
                                    lam_scale_,
                                    prng)
        else:
            raise NotImplementedError(
                    ("Only penalization = 'subsampling', "
                     "'random', and 'fully-random' have "
                     "been implemented. Found {}.".format(penalization)))

        # new instance of estimator
        new_estimator = clone(estimator)
        if boot_lam is not None:
            new_estimator.set_params(**{
                penalty_name: boot_lam,
            })

        # fit estimator
        num_subsamples = int(subsample * n_samples)
        rp = bootstrap(n_samples, num_subsamples, prng)
        new_estimator.fit(local_X[rp, :])

        # check that new_estimator.precision_ is real
        # if not, skip this boot_lam and try again
        if isinstance(new_estimator.precision_, list):
            prec_real_bools = []
            for prec in new_estimator.precision_:
                prec_real_bools.append(np.all(np.isreal(prec)))

            prec_is_real = np.all(np.array(prec_real_bools) is True)

        elif isinstance(new_estimator.precision_, np.ndarray):
            prec_is_real = np.all(np.isreal(new_estimator.precision_))

        else:
            raise ValueError("Estimator returned invalid precision_.")

    return index, (boot_lam, rp, new_estimator)


def _cpu_map(fun, param_grid, n_jobs, verbose=True):
    return Parallel(
        n_jobs=n_jobs,
        verbose=verbose,
        backend='threading',  # any sklearn backend should work here
    )(
        delayed(fun)(
            params
        )
        for params in param_grid)


def _spark_map(fun, indexed_param_grid, sc, seed, X_bc):
    '''We cannot pass a RandomState instance to each spark worker since it will
    behave identically across partitions.  Instead, we explictly handle the
    partitions with a newly seeded instance.

    The seed for each partition will be the "seed" (MonteCarloProfile.seed) +
    "split_index" which is the partition index.

    Following this trick:
        https://wegetsignal.wordpress.com/2015/05/08/
                generating-random-numbers-for-rdd-in-spark/
    '''
    def _wrap_random_state(split_index, partition):
        prng = np.random.RandomState(seed + split_index)
        yield map(partial(fun, prng=prng, X=X_bc), partition)

    par_param_grid = sc.parallelize(indexed_param_grid)
    indexed_results = par_param_grid.mapPartitionsWithIndex(
        _wrap_random_state).collect()
    return [item for sublist in indexed_results for item in sublist]


class ModelAverage(BaseEstimator):
    """
    Randomized model averaging meta-estimator.

    See analogous sklearn.linear_model.BaseRandomizedLinearModel.

    Parameters
    -----------
    estimator : An inverse covariance estimator instance
        After being fit, estimator.precision_ must either be a matrix with the
        precision or a list of precision matrices (e.g., path mode).

    n_trials : int (default=100)
        Number of random subsets for which to bootstrap the data.

    subsample : float in range (0, 1) (default=0.3)
        Fraction of examples to subsample in each bootstrap trial.

    normalize : bool (default=True)
        Determines whether the proportion_ matrix should be normalized to have
        values in the range (0, 1) or should be absolute.

    lam : float (default=0.5)
        Scalar lambda penalty used in penalization='random' mode.  Will be
        ignored in all other modes.

    lam_perturb : float \in (0, 1) (default=0.5)
        Scalar perturbation parameter used in penalization='random'.  Will be
        ignored in all other modes.

    penalization : one of 'subsampling', 'random' (default), 'fully-random'
        Strategy for generating new random penalization in each trial.

        subsampling: Only the observations will be subsampled, the original
                     penalty supplied in the estimator instance will be used.
                     Use this technique when the estimator does not support
                     matrix penalization (e.g., sklearn GraphLasso).

        random: In addition to randomly subsampling the observations, 'random'
                applies a randomly-perturbed 'lam' weight matrix.  The entries
                of the matrix take the value
                {lam * lam_perturb, lam / lam_perturb} with probability 1/2.
                User must supply a scalar 'lam' and 'lam_perturb' parameters.

        fully-random: In addition to randomly subsampling the observations,
                      'fully-random' generates a symmetric Gaussian matrix
                      appropriately scaled for the data.

        For more information on 'random' penalization, see:
            "Stability Selection"
            N. Meinhausen and P. Buehlmann, May 2009

            "Random lasso"
            S. Wang, B. Nan, S. Rosset, and J. Zhu, Apr 2011

        For more information on 'fully-random', see:
            "Mixed effects models for resampled network statistics improves
            statistical power to find differences in multi-subject functional
            connectivity"
            M. Narayan and G. Allen, March 2016

    support_thresh : float (0, 1)
        Threshold for estimating supports from proportions.  This is provided
        for convience.

    penalty_name : string (default='lam')
        Name of the penalty kwarg in the estimator.  This parameter is
        unimportant if penalization='subsampling'.

    bootstrap : callable fun (default=_default_bootstrap)
        A function that takes n_samples, num_subsamples as inputs and returns
        a list of sample indices in the range (0, n_samples-1).
        By default, indices are uniformly subsampled.

    n_jobs: int (optional)
        number of jobs to run in parallel (default 1).

    sc: sparkContext (optional)
        If a sparkContext object is provided, n_jobs will be ignore and the
        work will be parallelized via spark.

    seed : np.random.RandomState starting seed. (default=2)

    Attributes
    ----------
    proportion_ : matrix of size (n_features, n_features)
        Each entry indicates the sample probability (or count) of whether the
        inverse covariance is non-zero.

    support_ : matrix of size (n_features, n_features)
        Support estimate via thresholding proportions by support_thresh.

    estimators_ : list of estimator instances (n_trials, )
        The estimator instance from each trial.
        This returns an empty list if use_cache=False.

    lams_ : list of penalization matrices (n_trials, )
        The penalization matrix chosen in each trial.
        This returns an empty list if penalization='subsampling'.

    subsets_ : list of subset indices (n_trials, )
        The example indices chosen in each trial.
        This returns an empty list if use_cache=False.

    lam_ : float
        Average matrix value used among lam_ for all estimators.
    """
    def __init__(self, estimator=None, n_trials=100, subsample=0.3,
                 normalize=True, lam=0.5, lam_perturb=0.5,
                 penalization='random', penalty_name='lam', support_thresh=0.5,
                 bootstrap=_default_bootstrap, n_jobs=1, sc=None, seed=1):
        self.estimator = estimator
        self.n_trials = n_trials
        self.subsample = subsample
        self.normalize = normalize
        self.lam = lam
        self.lam_perturb = lam_perturb
        self.penalization = penalization
        self.penalty_name = penalty_name
        self.support_thresh = support_thresh
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.sc = sc
        self.seed = seed
        self.prng = np.random.RandomState(seed)

        self.proportion_ = None
        self.support_ = None
        self.lam_ = None
        self.lam_scale_ = None
        self.estimators_ = []
        self.lams_ = []
        self.subsets_ = []

        # default to QuicGraphLasso
        if self.estimator is None:
            self.estimator = QuicGraphLasso()

        if self.penalization != 'subsampling' and\
                not hasattr(self.estimator, self.penalty_name):
            raise ValueError(("Must specify valid penalty for "
                              "estimator: {}.".format(self.penalty_name)))

    def fit(self, X, y=None):
        """Learn a model averaged proportion matrix for X.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the proportion matrix.
        """
        X = check_array(X, ensure_min_features=2, estimator=self)
        X = as_float_array(X, copy=False, force_all_finite=False)

        n_samples, n_features = X.shape
        _, self.lam_scale_ = _init_coefs(X, method='cov')

        fit_fun = partial(
            _fit,
            penalization=self.penalization, lam=self.lam,
            lam_perturb=self.lam_perturb, lam_scale_=self.lam_scale_,
            estimator=self.estimator, penalty_name=self.penalty_name,
            subsample=self.subsample, bootstrap=self.bootstrap
        )
        indexed_param_grid = [(nn,) for nn in range(self.n_trials)]

        if self.sc is None:
            results = _cpu_map(
                partial(fit_fun, X=X, prng=self.prng),
                indexed_param_grid,
                n_jobs=self.n_jobs
            )
        else:
            X_bc = self.sc.broadcast(X)
            results = _spark_map(
                fit_fun,
                indexed_param_grid,
                self.sc,
                self.seed,
                X_bc
            )
            X_bc.unpersist()

        self.estimators_ = [e for r, (l, s, e) in results]
        self.subsets_ = [s for r, (l, s, e) in results]
        self.lams_ = [l for r, (l, s, e) in results]

        # reduce
        self.lam_ = 0.0
        self.proportion_ = np.zeros((n_features, n_features))
        for new_estimator in self.estimators_:
            # update proportions
            if isinstance(new_estimator.precision_, list):
                for prec in new_estimator.precision_:
                    self.proportion_[np.nonzero(prec)] += 1.

            elif isinstance(new_estimator.precision_, np.ndarray):
                self.proportion_[np.nonzero(new_estimator.precision_)] += 1.

            else:
                raise ValueError("Estimator returned invalid precision_.")

            # currently, dont estimate self.lam_ if penalty_name is different
            if self.penalty_name == 'lam':
                self.lam_ += np.mean(new_estimator.lam_.flat)

        # estimate support locations
        threshold = self.support_thresh * self.n_trials
        self.support_ = np.zeros(self.proportion_.shape)
        self.support_[self.proportion_ > threshold] = 1.0

        self.lam_ /= self.n_trials
        if self.normalize:
            self.proportion_ /= self.n_trials

    @property
    def precision_(self):
        '''Convenience property to make compatible with AdaptiveGraphLasso.
        This is not a very good precision estimate.
        '''
        return self.support_

    @property
    def covariance_(self):
        '''Convenience property to make compatible with AdaptiveGraphLasso.
        This is not a very good covariance estimate.
        '''
        return np.linalg.inv(self.support_)
