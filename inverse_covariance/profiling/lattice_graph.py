import numpy as np 
from .graphs import Graph, lattice


class LatticeGraph(Graph):
    """Returns the adjacency matrix for a lattice network via .create().

    The resulting network is a Toeplitz matrix with random values summing 
    between -1 and 1 and zeros along the diagonal. 
    
    The range of the values can be controlled via the parameters low and high.  
    If random_sign is false, all entries will be negative, otherwise their sign
    will be modulated at random with probability 1/2.

    Parameters
    ----------- 
    random sign : bool (default=False)
        Randomly modulate each entry by 1 or -1 with probability of 1/2.

    low : float (0, 1) (default=0.3)
        Lower bound for np.random.RandomState.uniform before normalization.

    high : float (0, 1) > low (default=0.7)
        Upper bound for np.random.RandomState.uniform before normalization.

    seed : int
        Seed for np.random.RandomState seed. (default=1)
    """
    def __init__(self, random_sign=False, low=0.7, high=0.7, **kwargs):
        self.random_sign = random_sign
        self.low = low
        self.high = high
        super(LatticeGraph, self).__init__(**kwargs)

    def create(self, n_features, alpha):
        """Build a new graph.

        Each row has maximum edges of np.ceil(alpha * n_features).

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
        adjacency = lattice(self.prng, n_features, alpha,
                            random_sign=self.random_sign,
                            low=self.low,
                            high=self.high) 
        precision = self.to_precision(adjacency)
        covariance = self.to_covariance(precision)
        return covariance, precision, adjacency
