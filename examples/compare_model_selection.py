import sys
sys.path.append('..')

from inverse_covariance import (
    QuicGraphLassoCV, 
    QuicGraphLassoEBIC,
)
from inverse_covariance.profiling import AverageError
from matplotlib import pyplot as plt


n_features = 50
n_trials = 100
verbose = True
score_metric = 'frobenius'

ae = AverageError(
        model_selection_estimator=QuicGraphLassoEBIC(
            gamma=0.3,
            score_metric=score_metric),
        n_features=n_features,
        n_trials=n_trials,
        verbose=verbose,
    )
ae.fit()
ae.show()
plt.title('EBIC, gamma = 0.3')

ae = AverageError(
        model_selection_estimator=QuicGraphLassoEBIC(
            gamma=0.0,
            score_metric=score_metric),
        n_features=n_features,
        n_trials=n_trials,
        verbose=verbose,
    )
ae.fit()
ae.show()
plt.title('EBIC, gamma = 0')

ae = AverageError(
        model_selection_estimator=QuicGraphLassoCV(
            score_metric=score_metric),
        n_features=n_features,
        n_trials=n_trials,
        verbose=verbose,
    )
ae.fit()
ae.show()
plt.title('CV')


raw_input()