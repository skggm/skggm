import numpy as np
from sklearn.base import BaseEstimator 


class ModelAverage(BaseEstimator):
    """
    Randomized model averaging meta-estimator.

    Parameters
    -----------        
    estimator : 
        After being fit, estimator.precision_ must either be a matrix with the 
        precision or a list of precision matrices (e.g., path mode).
        This should be compatible with QuicGraphLasso, QuicGraphLassoCV, as well
        as the scikit-learn variants.

    estimator_args :

    num_trials :

    subsample : float

    normalize :

    penalization : one of 'random', 'adaptive' 


    Attributes
    ----------
    proportion_ : 

    estimators_ : 

    lams_ : 
    
    """
    def __init__(self, estimator=None, estimator_args={}, num_trials=100, 
                 normalize=True, penalization='random', subsample=0.3):
        self.estimator = estimator 
        self.estimator_args = estimator_args
        self.num_trials = num_trials
        self.normalize = normalize
        self.penalization = penalization
        self.subsample = subsample

        self.proportion_ = None
        self.estimators_ = []
        self.lams_ = []

    def _random_weights(self, n_features):
        """Generate a symetric random matrix with ones along the diagonal.
        """
        weights = np.eye(n_features)
        n_off_diag = (n_features ** 2 - n_features) / 2 
        weights[np.triu_indices(n_features, k=1)] = np.random.randn(n_off_diag)
        weights = weights + weights.T - np.diag(weights.diagonal())
        return weights

    #def _adaptive_weights(self):
    
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
                lam = self._random_weights()
            else:
                raise NotImplementedError(
                    "Only penalization='random' has been implemented.")

            new_estimator = self.estimator(**self.estimator_args)

            num_subsamples = int(self.subsample * n_samples)
            rp = np.random.permutation(n_samples)[:num_subsamples]
            new_estimator.fit(X[rp, :])

            # QUESTION:  This updates the *nonzero* locations, do we only want the
            #            zero locations?  It's an easy change
            if isinstance(new_estimator.precision_, list):
                for prec in new_estimator.precision_:
                    self.proportion_[np.nonzero(prec)] += 1.
            elif isinstance(new_estimator.precision_, np.array):
                self.proportion_[np.nonzero(new_estimator.precision)] += 1.
            else:
                raise ValueError("Estimator returned invalid precision_.")

            self.estimators_.append(new_estimator)
            self.lams_.append(lam)

        if self.normalize:
            self.proportion_ /= self.num_trials


