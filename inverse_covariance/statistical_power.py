import numpy as np 

'''
Notes to self:

Similar the other meta estimators, this will take an estimator as a paramater 
(and kwargs), and then generate a bunch of random examples, run the estimator
and produce a statistical power plot (probability of recovering correct support
or very low error) as a function of 

    n, p, k

This is its own utility, does not make sense to live on InverseCovarianceEstimator
'''

class StatisticalPower(object):
    """
    """
    def __init__(self):
        pass