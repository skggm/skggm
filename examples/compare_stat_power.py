import sys
sys.path.append('..')

from inverse_covariance import (
    StatisticalPower,
    QuicGraphLassoCV, 
    QuicGraphLassoEBIC,
)
from matplotlib import pyplot as plt


'''
A few comments to do on phone with M:
- why do I still need to threshold, and by how much?
- what about that ebic thing that fixed the situation?
- In the case of model-average, do we threshold the proportion matrix to 
  determine a sparsity?  How does that work?
'''


'''
# EBIC, gamma=0.3
sp = StatisticalPower(
        model_selection_estimator=QuicGraphLassoEBIC(gamma=0.3),
        n_features=50,
        n_trials=100,
        verbose=True,
    )
sp.fit()
sp.show()
plt.title('gamma = 0.3')
sp = StatisticalPower(
        model_selection_estimator=QuicGraphLassoEBIC(gamma=0.0),
        n_features=50,
        n_trials=100,
        verbose=True,
    )
sp.fit()
sp.show()
plt.title('gamma = 0')
'''


# GraphLassoCV CV, 
sp = StatisticalPower(
        model_selection_estimator=QuicGraphLassoCV(),
        n_features=50,
        n_trials=100,
        n_jobs=2,
        verbose=True,
    )
sp.fit()
sp.show()


raw_input()