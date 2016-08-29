import sys
sys.path.append('..')

from inverse_covariance import (
    QuicGraphLassoCV, 
    QuicGraphLassoEBIC,
    ModelAverage
)
from inverse_covariance.profiling import AverageError 
from matplotlib import pyplot as plt


n_features = 50 
n_trials = 100
verbose = True

# average plots for QuicGraphLassoEBIC
ae = AverageError(
        model_selection_estimator=QuicGraphLassoEBIC(
            gamma=0.0),
        n_features=n_features,
        n_trials=n_trials,
        verbose=verbose,
    )
ae.fit()
ae.show()
plt.suptitle('QuicGraphLassoEBIC (BIC)')

# average plots for QuicGraphLassoCV
ae = AverageError(
        model_selection_estimator=QuicGraphLassoCV(),
        n_features=n_features,
        n_trials=n_trials,
        verbose=verbose,
    )
ae.fit()
ae.show()
plt.suptitle('QuicGraphLassoCV')

# average plots for ModelAverage + CV
ae = AverageError(
        model_selection_estimator=ModelAverage(
            n_trials=20,
            penalization='random',
            lam=0.2),
        n_features=n_features,
        n_trials=n_trials,
        verbose=verbose,
    )
ae.fit()
ae.show()
plt.suptitle('ModelAverage CV')

raw_input()