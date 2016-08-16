import numpy as np
from sklearn.base import BaseEstimator 


class ModelAverage(BaseEstimator):
    """
    Randomized model averaging meta-estimator.

    
    Parameters
    -----------        
    estimator : An inverse covariance estimator class
        After being fit, estimator.precision_ must either be a matrix with the 
        precision or a list of precision matrices (e.g., path mode).
        Important: The estimator must be able to take a *matrix* penalty,
                   such as 'lam' in QuicGraphLasso.
                   Set the penalty kwarg name using penalty='penalty_name'.

    estimator_args : A kwargs dict for estimator
        Each new instance of estimator will use these params.

    num_trials : int (default=100)
        Number of random subsets for which to bootstrap the data.

    use_cache : bool (default=True)
        If false, will optionally not cache each estimator instance and 
        penalization instance (to save memory).

    subsample : float in range (0, 1) (default=0.3)
        Fraction of examples to subsample in each bootstrap trial.

    normalize : bool (default=True)
        Determines whether the proportion_ matrix should be normalized to have
        values in the range (0, 1) or should be absolute.

    penalty : string
        Name of the penalty kwarg in the estimator
        e.g., 'lam' for QuicGraphLasso

    penalization : one of 'random', 'adaptive' 
        Strategy for generating new random penalization in each trial.

        For more information on 'random' penalization, see:
            "Stability Selection"
            N. Meinhausen and P. Buehlmann, May 2009

        For more information on 'random adaptive penalization', see:
            "Mixed effects models for resampled network statistics improves
            statistical power to find differences in multi-subject functional
            connectivity" 
            M. Narayan and G. Allen, March 2016


    Attributes
    ----------
    proportion_ : matrix of size (n_features, n_features)
        Each entry indicates the sample probability (or count) of whether the 
        inverse covariance is non-zero.

    estimators_ : list of estimator instances (num_trials, )
        The estimator instance from each trial.  
        This returns an empty list if use_cache=False.

    lams_ : list of penalization matrices (num_trials, )
        The penalization matrix chosen in each trial.
        This returns an empty list if use_cache=False.
    
    subsets_ : list of subset indices (num_trials, )
        The example indices chosen in each trial.
        This returns an empty list if use_cache=False.
    """
    def __init__(self, estimator=None, estimator_args={}, num_trials=100, 
                 normalize=True, penalization='random', subsample=0.3,
                 use_cache=True, penalty='lam'):
        self.estimator = estimator 
        self.estimator_args = estimator_args
        self.num_trials = num_trials
        self.normalize = normalize
        self.penalization = penalization
        self.subsample = subsample
        self.use_cache = use_cache
        self.penalty = penalty

        self.proportion_ = None
        self.estimators_ = []
        self.lams_ = []
        self.subsets_ = []

        if self.estimator is None:
            raise ValueError("ModelAvergae must be instantiated with an ",
                             "estimator.")


    def _random_weights(self, n_features):
        """Generate a symmetric random matrix with ones along the diagonal.
        """
        weights = np.eye(n_features)
        n_off_diag = (n_features ** 2 - n_features) / 2 
        weights[np.triu_indices(n_features, k=1)] = np.random.randn(n_off_diag)
        weights = weights + weights.T - np.diag(weights.diagonal())
        return weights


    # def _adaptive_weights(self): # stub


    def fit(self, X, y=None):
        """Learn a model averaged proportion matrix for X.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the proportion matrix.
        """
        n_samples, n_features = X.shape
        self.proportion_ = np.zeros((n_features, n_features))
        for nn in range(self.num_trials):
            if self.penalization == 'random':
                lam = self._random_weights(n_features)
            else:
                raise NotImplementedError(
                    "Only penalization='random' has been implemented.")

            self.estimator_args.update({
                self.penalty: lam, 
            })
            new_estimator = self.estimator(**self.estimator_args)

            num_subsamples = int(self.subsample * n_samples)
            rp = np.random.permutation(n_samples)[:num_subsamples]
            new_estimator.fit(X[rp, :])

            if isinstance(new_estimator.precision_, list):
                for prec in new_estimator.precision_:
                    self.proportion_[np.nonzero(prec)] += 1.
            elif isinstance(new_estimator.precision_, np.ndarray):
                self.proportion_[np.nonzero(new_estimator.precision_)] += 1.
            else:
                raise ValueError("Estimator returned invalid precision_.")

            if self.use_cache:
                self.estimators_.append(new_estimator)
                self.lams_.append(lam)
                self.subsets_.append(rp)

        if self.normalize:
            self.proportion_ /= self.num_trials


