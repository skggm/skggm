import sys
import pprint
import numpy as np
from sklearn.grid_search import GridSearchCV
sys.path.append('../inverse_covariance')
from inverse_covariance import InverseCovariance 


'''
Example of brute-force parameter search with InverseCovariance
'''

def estimate_parameters(n, p, num_folds, metric='log_likelihood'):
    # random normal matrix as input example
    X = np.random.randn(n, p)

    # define a cross-validation search grid 
    # lambda should vary from .01 or .001 to max off-diagonal value
    # note that when using initializ_method='cov', lambda will be scaled by max
    # off-diagonal value, so we use lam max = 1.0.
    search_grid = {
      'lam': np.logspace(np.log10(0.001), np.log10(1.0), num=5, endpoint=True),
      'path': [np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])],
      'mode': ['path'],
      'initialize_method': ['cov'],
      'metric': [metric],
    }

    # search for best parameters
    estimator = GridSearchCV(InverseCovariance(),
                            search_grid,
                            cv=num_folds,
                            refit=True,
                            verbose=1)
    estimator.fit(X)

    #print 'Best lambda path scale {}'.format(estimator.best_estimator_.score_best_path_scale_)
    print 'Best parameters:'
    pprint.pprint(estimator.best_params_)
    print 'Best score: {}'.format(estimator.score(X))


if __name__ == "__main__":
    p = 20 #200
    n = 10 #100
    num_folds = 2

    # fit with log_likelihood
    estimate_parameters(n, p, num_folds, metric='log_likelihood')

    # fit with kl-divergence
    estimate_parameters(n, p, num_folds, metric='kl')    