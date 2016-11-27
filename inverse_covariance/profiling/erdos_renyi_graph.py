from __future__ import absolute_import

import numpy as np 
from sklearn.datasets import make_sparse_spd_matrix
from .graphs import Graph 


class ErdosRenyiGraph(Graph):
    """Returns the adjacency matrix for Erdos-Renyi network via .create().

    Parameters
    ----------- 
    spd_low : float (0, 1)
        Equivalent to make_sparse_spd_matrix `smallest_coef`

    spd_high : float (0, 1)
        Equivalent to make_sparse_spd_matrix `largest_coef`

    seed : int
        Seed for np.random.RandomState seed. (default=1)
    """
    def __init__(self, spd_low=0.7, spd_high=0.7, **kwargs):
        self.spd_low = spd_low
        self.spd_high = spd_high
        super(ErdosRenyiGraph, self).__init__(**kwargs)

    def create(self, n_features, alpha):
        """Build a new graph.

        Parameters
        -----------        
        n_features : int 

        alpha : float (0,1) 
            # TODO: Better comment on this parameter.
            The complexity / sparsity factor.
        
        Returns
        -----------  
        (n_features, n_features) matrices: covariance, precision, adjacency
        """
        adjacency = make_sparse_spd_matrix(n_features,
                                           alpha=np.abs(1.0 - alpha), 
                                           smallest_coef=self.spd_low,
                                           largest_coef=self.spd_high,
                                           random_state=self.prng)

        precision = self.to_precision(adjacency)
        covariance = self.to_covariance(precision)
        return covariance, precision, adjacency
