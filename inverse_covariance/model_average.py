import numpy as np
from sklearn.base import BaseEstimator, clone


class ModelAverage(BaseEstimator):
    """
    Randomized model averaging meta-estimator.

    See analogous sklearn.linear_model.BaseRandomizedLinearModel.
    
    Parameters
    -----------        
    estimator : An inverse covariance estimator instance
        After being fit, estimator.precision_ must either be a matrix with the 
        precision or a list of precision matrices (e.g., path mode).
        Important: The estimator must be able to take a *matrix* penalty,
                   such as 'lam' in QuicGraphLasso.
                   Set the penalty kwarg name using penalty='penalty_name'.

    n_trials : int (default=100)
        Number of random subsets for which to bootstrap the data.

    use_cache : bool (default=True)
        If false, will optionally not cache each estimator instance and 
        penalization instance (to save memory).

    subsample : float in range (0, 1) (default=0.3)
        Fraction of examples to subsample in each bootstrap trial.

    normalize : bool (default=True)
        Determines whether the proportion_ matrix should be normalized to have
        values in the range (0, 1) or should be absolute.

    penalty : string (default='lam')
        Name of the penalty kwarg in the estimator.  This parameter is 
        unimportant if use_scalar_penalty=True.
        e.g., 'lam' for QuicGraphLasso, 'alpha' for GraphLasso

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

    use_scalar_penalty : bool (default=False)
        Set this to true if the graph lasso estimator does not support matrix
        penalization.  This leave penalization untouched and only bootstrap the
        samples.


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
        This returns an empty list if use_cache=False and/or 
        use_scalar_penalty=True
    
    subsets_ : list of subset indices (n_trials, )
        The example indices chosen in each trial.
        This returns an empty list if use_cache=False.
    """
    def __init__(self, estimator=None, n_trials=100, normalize=True,
                 penalization='random', subsample=0.3, use_cache=True,
                 penalty='lam', use_scalar_penalty=False):
        self.estimator = estimator 
        self.n_trials = n_trials
        self.normalize = normalize
        self.penalization = penalization
        self.subsample = subsample
        self.use_cache = use_cache
        self.penalty = penalty
        self.use_scalar_penalty = use_scalar_penalty

        self.proportion_ = None
        self.estimators_ = []
        self.lams_ = []
        self.subsets_ = []

        if self.estimator is None:
            raise ValueError("ModelAvergae must be instantiated with an ",
                             "estimator.")

        if not self.use_scalar_penalty and not hasattr(self.estimator, self.penalty):
            raise ValueError("Must specify valid penalty for estimator: {}.".format(
                self.penalty))


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
        for nn in range(self.n_trials):
            lam = None
            if not self.use_scalar_penalty:
                if self.penalization == 'random':
                    lam = self._random_weights(n_features)
                else:
                    raise NotImplementedError(
                        "Only penalization='random' has been implemented.")


            # new instance of estimator
            new_estimator = clone(self.estimator)
            
            # patch estimator args with new penalty
            if lam is not None:
                new_estimator.set_params(**{
                    self.penalty: lam,
                }) 

            # fit estimator
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
                self.subsets_.append(rp)
                if not self.use_scalar_penalty:
                    self.lams_.append(lam)

        if self.normalize:
            self.proportion_ /= self.n_trials


