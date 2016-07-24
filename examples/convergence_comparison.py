import sys
sys.path.append('../inverse_covariance')

from sklearn.covariance import graph_lasso
from inverse_covariance import quic
import numpy as np

'''
Demonstration of cases where GraphLasso fails to converge and quic succeeds.

"The graphical lasso: New Insights and alternatives", by Mazumder & Hastie 2012.
https://web.stanford.edu/~hastie/Papers/glassoinsights.pdf

TODO:
    - shall we use the sklearn objects instead of functions?
    - what do we want to output here?
'''

#############################################################################
# Example 1
# graph_lasso fails to converge at lam = .009 * np.max(np.abs(Shat))
X = np.loadtxt('data/Mazumder_example1.txt', delimiter=',')
Shat = np.cov(X, rowvar=0)
try:
    graph_lasso(Shat, alpha=.004)
except FloatingPointError as e:
    print('{0}'.format(e))
vals = quic(Shat, .004)


#############################################################################
# Example 2
# graph_lasso fails to converge at lam = .009 * np.max(np.abs(Shat))
X = np.loadtxt('data/Mazumder_example2.txt', delimiter=',')
Shat = np.cov(X, rowvar=0)
try:
    graph_lasso(Shat, alpha=.02)
except FloatingPointError as e:
    print('{0}'.format(e))
vals = quic(Shat, .02)