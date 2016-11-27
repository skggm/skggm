import numpy as np 
import scipy as sp


def lattice(prng, n_features, alpha, random_sign=False, low=0.3, high=0.7):
    """Returns the adjacency matrix for a lattice network. 
    
    The resulting network is a Toeplitz matrix with random values summing 
    between -1 and 1 and zeros along the diagonal. 
    
    The range of the values can be controlled via the parameters low and high.  
    If random_sign is false, all entries will be negative, otherwise their sign
    will be modulated at random with probability 1/2.

    Each row has maximum edges of np.ceil(alpha * n_features).

    Parameters
    -----------        
    n_features : int 

    alpha : float (0, 1) 
        The complexity / sparsity factor.

    random sign : bool (default=False)
        Randomly modulate each entry by 1 or -1 with probability of 1/2.

    low : float (0, 1) (default=0.3)
        Lower bound for np.random.RandomState.uniform before normalization.

    high : float (0, 1) > low (default=0.7)
        Upper bound for np.random.RandomState.uniform before normalization.
    """
    adj = np.zeros((n_features, n_features))
    row = np.zeros((n_features,))
    degree = int(1 + np.round(alpha * n_features / 2.))

    if random_sign:
        sign_row = (-1.0 * np.ones(degree) + 
                    2 * (prng.uniform(low=0, high=1, size=degree) > .5))
    else:
        # QUESTION to M:
        # Why is the default all negatives?
        sign_row = -1.0 * np.ones(degree)
    
    # populate shifted by 1 to avoid diagonal
    row[1: 1 + degree] = sign_row * prng.uniform(low=low, high=high, size=degree)

    # QUESTION to M:
    # Why is this normalized this way (can sum between 1 and -1)?
    # What if the row sums to 0 (unlikely, but could be)? 
    row /= np.abs(np.sum(row))  

    return sp.linalg.toeplitz(c=row, r=row)


def blocks(prng, block, n_blocks=2, chain_blocks=True):
    """Replicates `block` matrix n_blocks times diagonally to create a
    square matrix of size n_features = block.size[0] * n_blocks and with zeros
    along the diagonal.

    The graph can be made fully connected using chaining assumption when 
    chain_blocks=True (default).
    
    This utility can be used to generate cluster networks or banded lattice n
    networks, among others.

    Parameters
    -----------        
    block : 2D array (n_block_features, n_block_features) 
        Prototype block adjacency matrix.

    n_blocks : int (default=2)
        Number of blocks.  Returned matrix will be square with 
        shape n_block_features * n_blocks.

    chain_blocks : bool (default=True)
        Apply random lattice structure to chain blocks together.
    """
    n_block_features, _ = block.shape
    n_features = n_block_features * n_blocks
    adjacency = np.zeros((n_features, n_features))    
        
    dep_groups = np.eye(n_blocks)
    if chain_blocks:
        chain_alpha = np.round(0.01 + 0.5 / n_blocks, 2)
        chain_groups = lattice(prng, n_blocks, chain_alpha, random_sign=False)
        chain_groups *= -0.1
        dep_groups += chain_groups
        
    adjacency = np.kron(dep_groups, block)
    adjacency[np.where(np.eye(n_features))] = 0 
    return adjacency


def _to_diagonally_dominant(adjacency):
    """Make unweighted adjacency matrix M diagonally dominant using the 
    Laplacian.
    """    
    adjacency += np.diag(np.sum(adjacency != 0, axis=1) + 0.01)
    return adjacency


def _to_diagonally_dominant_weighted(adjacency):
    """Make weighted adjacency matrix M diagonally dominant using the 
    Laplacian.

    QUESTION:  Which is it?
    NOTE:  This was called make_diagonally_dominant() in examples
           but weighted in matlab version.
    """    
    adjacency += np.diag(np.sum(np.abs(adjacency), axis=1) + 0.01)
    return adjacency


def _to_correlation(adjacency):
    """Call only after diagonal dominance is ensured."""   
    d = np.sqrt(np.diag(adjacency))
    adjacency /= d
    adjacency /= d[:, np.newaxis]
    return adjacency


class Graph(object):
    """Base class that returns the adjacency matrix for a network via
    the .create() method.

    Parameters
    ----------- 
    seed : int
        Seed for np.random.RandomState seed. (default=1)
    """
    def __init__(self, seed=1):
        self.seed = seed
        self.prng = np.random.RandomState(self.seed)

    def to_precision(self, adjacency, weighted=True):
        if weighted:
            return _to_correlation(_to_diagonally_dominant_weighted(adjacency))
        else:
            return _to_correlation(_to_diagonally_dominant(adjacency))

    def to_covariance(self, precision):
        return _to_correlation(np.linalg.inv(precision))

    def create(self, n_features, alpha):
        """Build a new graph.

        Parameters
        -----------        
        n_features : int 

        alpha : float (0,1) 
            The complexity / sparsity factor for each graph type.
        
        Returns
        -----------  
        (n_features, n_features) matrices: covariance, precision, adjacency 
        """
        pass
