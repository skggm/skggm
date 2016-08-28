import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_array, as_float_array

from .inverse_covariance import _init_coefs


def _check_psd(m):
    w, v = np.linalg.eig(m)
    return np.all(w >= 0)


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
    b_0 = berns == 0
    b_1 = berns == 1
    berns[b_0] = 1. * lam * lam_perturb
    berns[b_1] = 1. * lam / lam_perturb
    weights[np.triu_indices(n_features, k=1)] = berns
    weights[weights < 0] = 0
    weights = weights + weights.T
    return weights


def _generate_until_valid(weight_fun, *args):
    """Generate weight matrices until find a positive semi-definite one.
    """
    while True:
        new_weights = weight_fun(*args)
        if _check_psd(new_weights):
            return new_weights


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

    use_cache : bool (default=True)
        If false, will optionally not cache each estimator instance and 
        penalization instance (to save memory).

    penalty_name : string (default='lam')
        Name of the penalty kwarg in the estimator.  This parameter is 
        unimportant if penalization='subsampling'.


    Attributes
    ----------
    proportion_ : matrix of size (n_features, n_features)
        Each entry indicates the sample probability (or count) of whether the 
        inverse covariance is non-zero.

    estimators_ : list of estimator instances (n_trials, )
        The estimator instance from each trial.  
        This returns an empty list if use_cache=False.

    lams_ : list of penalization matrices (n_trials, )
        The penalization matrix chosen in each trial.
        This returns an empty list if penalization='subsampling'.
    
    subsets_ : list of subset indices (n_trials, )
        The example indices chosen in each trial.
        This returns an empty list if use_cache=False.
    """
    def __init__(self, estimator=None, n_trials=100, subsample=0.3, 
                 normalize=True, lam=0.5, lam_perturb=0.5, penalization='random',
                 use_cache=True, penalty_name='lam'):
        self.estimator = estimator 
        self.n_trials = n_trials
        self.subsample = subsample
        self.normalize = normalize
        self.lam = lam 
        self.lam_perturb = lam_perturb
        self.penalization = penalization
        self.use_cache = use_cache
        self.penalty_name = penalty_name

        self.proportion_ = None
        self.estimators_ = []
        self.lams_ = []
        self.subsets_ = []

        if self.estimator is None:
            raise ValueError("ModelAvergae must be instantiated with an "
                             "estimator.")

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
        
        self.proportion_ = np.zeros((n_features, n_features))
        for nn in range(self.n_trials):
            prec_is_real = False
            while not prec_is_real:
                lam = None
                if self.penalization == 'random':
                    lam = _generate_until_valid(_random_weights, n_features)
                elif self.penalization == 'full-random':
                    lam_scale = _init_coefs(X, method='cov')
                    lam = _generate_until_valid(_fully_random_weights,
                                                n_features,
                                                lam_scale)
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
                rp = np.random.permutation(n_samples)[:num_subsamples]
                new_estimator.fit(X[rp, :])

                # check that new_estimator.precision_ is real
                # if not, skip this lam and try again
                if isinstance(new_estimator.precision_, list):
                    prec_is_real = True
                    for prec in new_estimator.precision_:
                        prec_is_real *= np.all(np.isreal(prec))
                
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

            if self.use_cache:
                self.estimators_.append(new_estimator)
                self.subsets_.append(rp)
                if not self.use_scalar_penalty:
                    self.lams_.append(lam)

        if self.normalize:
            self.proportion_ /= self.n_trials


