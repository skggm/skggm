import sys
sys.path.append('..')

from inverse_covariance import (
    QuicGraphLassoCV, 
    QuicGraphLassoEBIC,
    ModelAverage
)
from inverse_covariance.profiling import AverageError 
from matplotlib import pyplot as plt


n_features = 50 # 50
n_trials = 5 #100
verbose = True

ae = AverageError(
        model_selection_estimator=ModelAverage(
            n_trials=20,
            penalization='random',
            subsample=0.9,
            lam=0.2),
        n_features=n_features,
        n_trials=n_trials,
        verbose=verbose,
    )
ae.fit()
ae.show()
plt.suptitle('Model Average')


'''
ae = AverageError(
        model_selection_estimator=QuicGraphLassoEBIC(
            gamma=0.0),
        n_features=n_features,
        n_trials=n_trials,
        verbose=verbose,
    )
ae.fit()
ae.show()
plt.suptitle('EBIC, gamma = 0')

ae = AverageError(
        model_selection_estimator=QuicGraphLassoCV(),
        n_features=n_features,
        n_trials=n_trials,
        verbose=verbose,
    )
ae.fit()
ae.show()
plt.suptitle('CV')
'''

raw_input()