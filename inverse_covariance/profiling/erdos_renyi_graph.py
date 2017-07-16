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

    n_blocks : int (default=2)
        Number of blocks.  Returned matrix will be square with
        shape n_block_features * n_blocks.

    chain_blocks : bool (default=True)
        Apply random lattice structure to chain blocks together.

    seed : int
        Seed for np.random.RandomState seed. (default=1)
    """
    def __init__(self, spd_low=0.7, spd_high=0.7, **kwargs):
        self.spd_low = spd_low
        self.spd_high = spd_high
        super(ErdosRenyiGraph, self).__init__(**kwargs)

    def prototype_adjacency(self, n_block_features, alpha):
        """Build a new graph.

        Doc for ".create(n_features, alpha)"

        Parameters
        -----------
        n_features : int

        alpha : float (0,1)
            The complexity / sparsity factor.
            This is (1 - alpha_0) in sklearn.datasets.make_sparse_spd_matrix
            where alpha_0 is the probability that a coefficient is zero.

        Returns
        -----------
        (n_features, n_features) matrices: covariance, precision, adjacency
        """
        return make_sparse_spd_matrix(n_block_features,
                                      alpha=np.abs(1.0 - alpha),
                                      smallest_coef=self.spd_low,
                                      largest_coef=self.spd_high,
                                      random_state=self.prng)
