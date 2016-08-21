import sys
sys.path.append('..')

from inverse_covariance import (
    GraphLassoSP,
    QuicGraphLassoCV, 
    QuicGraphLassoEBIC,
)

'''
sp = GraphLassoSP(
        model_selection_estimator=QuicGraphLassoEBIC,
        model_selection_estimator_args={
            'gamma': 0.1,
        },
        n_features=50,
        n_trials=100,
        verbose=True,
    )
'''

sp = GraphLassoSP(
        model_selection_estimator=QuicGraphLassoCV,
        n_features=30,
        n_trials=100,
        verbose=True,
    )

sp.fit()

sp.show()