from __future__ import absolute_import

import numpy as np 
from .graphs import Graph, lattice, blocks


class BandedGraph(Graph):
    """Returns the adjacency matrix for a banded network via .create().

    The graph can be made fully connected using chaining assumption when 
    chain_blocks=True (default).

    Parameters
    ----------- 
    random sign : bool (default=False)
        Randomly modulate each entry by 1 or -1 with probability of 1/2.

    low : float (0, 1) (default=0.3)
        Lower bound for np.random.RandomState.uniform before normalization.

    high : float (0, 1) > low (default=0.7)
        Upper bound for np.random.RandomState.uniform before normalization.

    n_blocks : int (default=2)
        Number of blocks.  Returned matrix will be square with 
        shape n_block_features * n_blocks.

    chain_blocks : bool (default=True)
        Apply random lattice structure to chain blocks together.

    seed : int
        Seed for np.random.RandomState seed. (default=1)
    """
    def __init__(self, random_sign=False, low=0.7, high=0.7, 
                 n_blocks=2, chain_blocks=True, **kwargs):
        self.random_sign = random_sign
        self.low = low
        self.high = high
        self.n_blocks = n_blocks
        self.chain_blocks = chain_blocks
        super(BandedGraph, self).__init__(**kwargs)

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
        covariance : 

        precision : 

        adjacency : 
        """
        n_block_features = int(np.floor(1. * n_features / self.n_blocks))

        block_adj = lattice(self.prng, n_block_features, alpha,
                            random_sign=self.random_sign,
                            low=self.low,
                            high=self.high) 

        adjacency = blocks(self.prng,
                           block_adj,
                           n_blocks=self.n_blocks,
                           chain_blocks=self.chain_blocks)

        precision = self.to_precision(adjacency)
        covariance = self.to_covariance(precision)
        return covariance, precision, adjacency
