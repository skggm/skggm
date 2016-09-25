import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_array, as_float_array

from .inverse_covariance import _init_coefs
from . import QuicGraphLassoCV


def _check_psd(m):
    return np.all(np.linalg.eigvals(m) >= 0)


def _fully_random_weights(n_features, lam_scale):
    """Generate a symmetric random matrix with zeros along the diagonal.
    """
    weights = np.zeros((n_features, n_features))
    n_off_diag = (n_features ** 2 - n_features) / 2 
    weights[np.triu_indices(n_features, k=1)] =\
            0.1 * lam_scale * np.random.randn(n_off_diag) + (0.25 * lam_scale)
    weights[weights < 0] = 0
    weights = weights + weights.T
    return weights


def _random_weights(n_features, lam, lam_perturb):
    """Generate a symmetric random matrix with zeros along the diagnoal and 
    non-zero elements take the value {lam * lam_perturb, lam / lam_perturb} 
    with probability 1/2.
    """
    weights = np.zeros((n_features, n_features))
    n_off_diag = (n_features ** 2 - n_features) / 2
    berns = np.random.binomial(1, 0.5, size=n_off_diag)
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
    off_diag_sums = np.sum(weights, axis=1) # NOTE: assumes diag is zero
    mod_mat = np.linalg.inv(np.sqrt(np.diag(off_diag_sums)))
    return np.dot(mod_mat, weights, mod_mat)


def _default_bootstrap(n_samples, num_subsamples):
    """Returns an array of integers (0, n_samples-1)^num_subsamples.
    """
    return np.random.permutation(n_samples)[:num_subsamples]


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

    use_cache : bool (default=True)
        If false, will optionally not cache each estimator instance and 
        penalization instance (to save memory).

    penalty_name : string (default='lam')
        Name of the penalty kwarg in the estimator.  This parameter is 
        unimportant if penalization='subsampling'.

    bootstrap : callable fun (default=_default_bootstrap)
        A function that takes n_samples, num_subsamples as inputs and returns 
        a list of sample indices in the range (0, n_samples-1).
        By default, indices are uniformly subsampled.

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
                 normalize=True, lam=0.5, lam_perturb=0.5, penalization='random',
                 use_cache=True, penalty_name='lam', support_thresh=0.5,
                 bootstrap=_default_bootstrap):
        self.estimator = estimator 
        self.n_trials = n_trials
        self.subsample = subsample
        self.normalize = normalize
        self.lam = lam 
        self.lam_perturb = lam_perturb
        self.penalization = penalization
        self.use_cache = use_cache
        self.penalty_name = penalty_name
        self.support_thresh = support_thresh
        self.bootstrap = bootstrap

        self.proportion_ = None
        self.support_ = None
        self.lam_ = None
        self.lam_scale_ = None
        self.estimators_ = []
        self.lams_ = []
        self.subsets_ = []

        # default to QuicGraphLassoCV
        if self.estimator is None:
            self.estimator = QuicGraphLassoCV()

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
        
        self.lam_ = 0.0
        self.proportion_ = np.zeros((n_features, n_features))
        for nn in range(self.n_trials):
            prec_is_real = False
            while not prec_is_real:
                lam = None
                if self.penalization == 'subsampling':
                    pass
                elif self.penalization == 'random':
                    lam = _fix_weights(_random_weights, 
                                       n_features,
                                       self.lam,
                                       self.lam_perturb)
                elif self.penalization == 'fully-random':
                    lam = _fix_weights(_fully_random_weights,
                                       n_features,
                                       self.lam_scale_)
                else:
                    raise NotImplementedError(
                            ("Only penalization = 'subsampling', "
                            "'random', and 'fully-random' have "
                            "been implemented."))

                # new instance of estimator
                new_estimator = clone(self.estimator)
                if lam is not None:
                    new_estimator.set_params(**{
                        self.penalty_name: lam,
                    }) 

                # fit estimator
                num_subsamples = int(self.subsample * n_samples)
                rp = self.bootstrap(n_samples, num_subsamples)
                new_estimator.fit(X[rp, :])

                # check that new_estimator.precision_ is real
                # if not, skip this lam and try again
                if isinstance(new_estimator.precision_, list):
                    prec_real_bools = []
                    for prec in new_estimator.precision_:
                        prec_real_bools.append(np.all(np.isreal(prec)))

                    prec_is_real = np.all(np.array(prec_real_bools) == True)
                
                elif isinstance(new_estimator.precision_, np.ndarray):
                    prec_is_real = np.all(np.isreal(new_estimator.precision_))

                else:
                    raise ValueError("Estimator returned invalid precision_.")

            # update proportions
            if isinstance(new_estimator.precision_, list):
                for prec in new_estimator.precision_:
                    self.proportion_[np.nonzero(prec)] += 1.
            
            elif isinstance(new_estimator.precision_, np.ndarray):
                self.proportion_[np.nonzero(new_estimator.precision_)] += 1.
            
            else:
                raise ValueError("Estimator returned invalid precision_.")

            # estimate support locations
            threshold = self.support_thresh * self.n_trials
            self.support_ = np.zeros(self.proportion_.shape)
            self.support_[self.proportion_ > threshold] = 1.0

            # currently, dont estimate self.lam_ if penalty_name is different
            if self.penalty_name == 'lam':
                self.lam_ += np.mean(new_estimator.lam_.flat)
            
            # save estimators, subsets, and lambdas
            if self.use_cache:
                self.estimators_.append(new_estimator)
                self.subsets_.append(rp)
                if lam is not None:
                    self.lams_.append(lam)

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
    
