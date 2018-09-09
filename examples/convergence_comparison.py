"""
Convergence Failure of Glasso
=============================

Demonstration of cases where graph_lasso fails to converge and quic succeeds.

"The graphical lasso: New Insights and alternatives", by Mazumder & Hastie 2012.
https://web.stanford.edu/~hastie/Papers/glassoinsights.pdf

"""

import sys

sys.path.append("..")
sys.path.append("../inverse_covariance")

from sklearn.covariance import graph_lasso
from inverse_covariance import quic
import numpy as np


#############################################################################
# Example 1
# graph_lasso fails to converge at lam = .009 * np.max(np.abs(Shat))
X = np.loadtxt("data/Mazumder_example1.txt", delimiter=",")
Shat = np.cov(X, rowvar=0)
try:
    graph_lasso(Shat, alpha=.004)
except FloatingPointError as e:
    print("{0}".format(e))
vals = quic(Shat, .004)


#############################################################################
# Example 2
# graph_lasso fails to converge at lam = .009 * np.max(np.abs(Shat))
X = np.loadtxt("data/Mazumder_example2.txt", delimiter=",")
Shat = np.cov(X, rowvar=0)
try:
    graph_lasso(Shat, alpha=.02)
except FloatingPointError as e:
    print("{0}".format(e))
vals = quic(Shat, .02)
