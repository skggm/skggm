import sys
sys.path.append('..')

from sklearn.covariance import GraphLassoCV
from inverse_covariance import (
    QuicGraphLassoCV, 
    QuicGraphLassoEBIC,
)
from inverse_covariance.profiling import (
    AverageError,
)
from matplotlib import pyplot as plt


ae = AverageError(
        model_selection_estimator=QuicGraphLassoEBIC(
            gamma=0.3,
            score_metric='frobenius'),
        n_features=50,
        n_trials=100,
        verbose=True,
    )
ae.fit()
ae.show()
plt.title('EBIC, gamma = 0.3')
ae = AverageError(
        model_selection_estimator=QuicGraphLassoEBIC(
            gamma=0.0,
            score_metric='frobenius'),
        n_features=50,
        n_trials=100,
        verbose=True,
    )
ae.fit()
ae.show()
plt.title('EBIC, gamma = 0')


'''
start = time.time()
sp = StatisticalPower(
        model_selection_estimator=QuicGraphLassoCV(),
        n_features=100,
        n_trials=100,
        n_jobs=1,
        verbose=True,
    )
sp.fit()
end = time.time()
print 'Elapsed time for single thread {}'.format(end - start)

start = time.time()
sp = StatisticalPower(
        model_selection_estimator=QuicGraphLassoCV(),
        n_features=100,
        n_trials=100,
        n_jobs=4,
        verbose=True,
    )
sp.fit()
end = time.time()
print 'Elapsed time for 4 thread {}'.format(end - start)
'''

'''
# QuicGraphLassoCV CV, 
sp = StatisticalPower(
        model_selection_estimator=QuicGraphLassoCV(),
        n_features=50,
        n_trials=100,
        n_jobs=4,
        verbose=True,
    )
sp.fit()
sp.show()
'''

'''
# GraphLassoCV CV, 
sp = StatisticalPower(
        model_selection_estimator=GraphLassoCV(),
        n_features=50,
        n_trials=100,
        verbose=True,
        penalty_='alpha_',
    )
sp.fit()
sp.show()
'''

raw_input()