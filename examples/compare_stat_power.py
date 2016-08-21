import sys
sys.path.append('..')

from inverse_covariance import (
    GraphLassoSP,
    QuicGraphLassoCV, 
    QuicGraphLassoEBIC,
)


'''
A few comments to do on phone with M:
- why do I still need to threshold, and by how much?
- what about that ebic thing that fixed the situation?
- what about the model-average, that doesn't give a lambda back, should I 
  be designing for that case?
'''


'''
# EBIC, gamma=0.3
sp = GraphLassoSP(
        model_selection_estimator=QuicGraphLassoEBIC,
        model_selection_estimator_args={
            'gamma': 0.3,
        },
        n_features=50,
        n_trials=10,
        verbose=True,
    )
'''


# GraphLassoCV CV, 
sp = GraphLassoSP(
        model_selection_estimator=QuicGraphLassoCV(),
        n_features=50,
        n_trials=10,
        verbose=True,
    )


sp.fit()

sp.show()
raw_input()