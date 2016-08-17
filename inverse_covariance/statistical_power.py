import numpy as np 

from sklearn.datasets import make_sparse_spd_matrix


'''
Notes to self:

Similar the other meta estimators, this will take an estimator as a paramater 
(and kwargs), and then generate a bunch of random examples, run the estimator
and produce a statistical power plot (probability of recovering correct support
or very low error) as a function of 

    n, p, k

This is its own utility, does not make sense to live on InverseCovarianceEstimator
'''

def _new_sample(n_samples, n_features, alpha):
    prng = np.random.RandomState(1)
    prec = make_sparse_spd_matrix(n_features,
                                  alpha=alpha, # prob that a coeff is nonzero
                                  smallest_coef=0.1,
                                  largest_coef=0.9,
                                  random_state=prng)
    cov = np.linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    return X, cov, prec

class StatisticalPower(object):
    """
    """
    def __init__(self, estimator=None, estimator_args={}, n_trials=100):
        self.estimator = estimator 
        self.estimator_args = estimator_args

    