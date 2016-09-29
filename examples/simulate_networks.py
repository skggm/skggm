import numpy as np
import scipy as sp
from scipy import linalg
from sklearn.base import clone
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.externals.joblib import Parallel, delayed
from matplotlib import pyplot as plt
import seaborn as sns

import inverse_covariance as ic
from inverse_covariance import (
    QuicGraphLasso,
    QuicGraphLassoCV,
    QuicGraphLassoEBIC,
    AdaptiveGraphLasso,
)

def new_graph(n_features, alpha, adj_type='erdos-renyi', random_sign=False,seed=1):
    prng = np.random.RandomState(seed)
    
    if adj_type=='erdos-renyi':
        adjacency = random_er_network(n_features=n_features,alpha=abs(1.0-alpha),random_state=prng)
    elif adj_type=='banded':
        adjacency = lattice_network(n_features=n_features,alpha=alpha,random_sign=random_sign,random_state=prng)
    elif adj_type=='cluster':
        if n_features<=25:
            n_groups=3
        else:
            n_groups=5            
        adjacency = cluster_network(n_features=n_features,n_groups=n_groups,alpha=alpha,adj_type='cluster',random_sign=random_sign,random_state=prng)
    else:
        adjacency = random_er_network(n_features=n_features,alpha=alpha,random_state=prng)

    prec = make_diag_dominant(adjacency)
    prec = make_correlation(prec)
    cov = np.linalg.inv(prec)
    cov = make_correlation(cov)
    return cov, prec, adjacency

def random_er_network(n_features, alpha,random_state=np.random.RandomState(1)):
    adj = make_sparse_spd_matrix(n_features,
                                  alpha=alpha, # prob that a coeff is zero
                                  smallest_coef=0.7,
                                  largest_coef=0.7,
                                  random_state=random_state)
    return adj

def lattice_network(n_features, alpha =.3, random_sign=False,random_state=np.random.RandomState(1)):
    """
    Creates a lattice network of size n_features x n_features. Each row has maximum edges of ceil(.3*n_features)
    """
    
    adj = np.zeros([n_features,n_features])
    row = np.zeros([n_features])
    degree = 1+np.round(alpha*n_features/2)
    if random_sign:
        sign_row = -1.0*np.ones(degree)+ 2*(random_state.uniform(low=0,high=1,size=int(degree))>.5)
    else:
        sign_row = -1.0*np.ones(degree)
    row[1:1+degree] = sign_row*random_state.uniform(low=.3, high=.7, size=int(degree))
    row /= abs(sum(row))
    adj = sp.linalg.toeplitz(c=row,r=row)
    return adj

def hub_network(n_features, alpha,random_state=np.random.RandomState(1)):
    
    adj = np.zeros([n_features,n_features])
        
    return adj

def smallw_network(n_features, alpha,random_state=np.random.RandomState(1)):
    """
    Create disjoint groups of variables (e.g n_features=15, 5 groups of size 3). 
    Make fully connected using chain assumption.  
    """
    adj = np.zeros([n_features,n_features])
        
    return adj

def cluster_network(n_features, n_groups, alpha, random_sign=False,adj_type='banded',chain_blocks=True,random_state=np.random.RandomState(1)):
    """
    Create disjoint groups of variables (e.g n_features=15, 5 groups of size 3). 
    Make fully connected using chaining assumption.  
    """
    adj = np.zeros([n_features,n_features])    
    n_block = int(np.floor(n_features/n_groups))
    
    if adj_type=='banded':
        small_adj = lattice_network(n_features=n_block,alpha=alpha,random_sign=random_sign,random_state=random_state)
    elif adj_type=='cluster':
        small_adj = -np.ones((n_block,n_block))*.5 + random_state.uniform(low=.05, high=.2, size=(n_block,n_block))
    else:
        small_adj = lattice_network(n_features=n_block,alpha=alpha,random_sign=random_sign,random_state=random_state) 
    
    if chain_blocks:
        dep_groups = lattice_network(n_features=n_groups,alpha=round(.01+0.5/n_groups,2),random_sign=False,random_state=random_state)
        dep_groups *= -.1
        dep_groups += np.eye(n_groups)
    else:
        dep_groups = np.eye(n_groups)
        
    adj = np.kron(dep_groups, small_adj)
    adj[np.where(np.eye(n_features))] = 0
    
    return adj


def make_diag_dominant(adjacency):    
    
    d = np.diag(np.sum(np.abs(adjacency),axis=1)+.01)
    adjacency += d
    return adjacency

def make_correlation(adjacency):
    """
    Call only after diagonal dominance is ensured. 
    TODO: Check for diagonally dominant adjacency first. 
    """   
    d = np.sqrt(np.diag(adjacency))
    adjacency /= d
    adjacency /= d[:, np.newaxis]
    return adjacency

# Simulate multivariate normal and check empirical covariances
def mvn(n_samples, n_features, cov, random_state=np.random.RandomState(2)):
    X = random_state.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    return X


# MCMC Statistical Power
def _plot_spower(results, grid, ks):
    plt.figure()
    plt.plot(grid, results.T, lw=2)
    plt.xlabel('n/p (n_samples / n_features)')
    plt.ylabel('P(exact recovery)')
    legend_text = []
    for ks in ks:
        legend_text.append('sparsity={}'.format(ks))
    plt.legend(legend_text)
    plt.show()


def _exact_support(prec, prec_hat):
    # Q: why do we need something like this?, and why must eps be so big?
    # Q: can we automatically determine what this threshold should be?    
    eps = .2
    #eps = np.finfo(prec_hat.dtype).eps # too small
    prec_hat[np.abs(prec_hat) <= eps] = 0.0
    
    not_empty = np.count_nonzero(np.triu(prec,1))>0
    
    return not_empty & np.array_equal(
            np.nonzero(np.triu(prec,1).flat)[0],
            np.nonzero(np.triu(prec_hat,1).flat)[0])


def _sp_trial(trial_estimator, n_samples, n_features, cov, adj, random_state):
    X = mvn(n_samples, n_features, cov,random_state=random_state)
    new_estimator = clone(trial_estimator)
    new_estimator.fit(X)
    return _exact_support(adj, new_estimator.precision_)


class StatisticalPower(object):
    """Compute the statistical power P(exact support) of a model selector for
    different values of alpha over grid of n_samples / n_features.

    For each choice of alpha, we select a fixed test graph.
    
    For each choice of n_samples / n_features, we learn the model selection
    penalty just once and apply this learned value to each subsequent random
    trial (new instances with the fixed covariance).

    Parameters
    -----------        
    model_selection_estimator : An inverse covariance estimator instance 
        This estimator must be able to select a penalization parameter. 
        Use .penalty_ to obtain selected penalty.

    n_features : int (default=50)
        Fixed number of features to test.

    n_trials : int (default=100)
        Number of examples to draw to measure P(recovery).

    trial_estimator : An inverse covariance estimator instance (default=None)
        Estimator to use on each instance after selecting a penalty lambda.
        If None, this will use QuicGraphLasso with lambda obtained with 
        model_selection_estimator.
        Use .penalty to set selected penalty.

    penalty_ : string (default='lam_')
        Name of the selected best penalty in estimator
        e.g., 'lam_' for QuicGraphLassoCV, QuicGraphLassoEBIC,
              'alpha_' for GraphLassoCV

    penalty : string (default='lam')
        Name of the penalty kwarg in the estimator.  
        e.g., 'lam' for QuicGraphLasso, 'alpha' for GraphLasso

    n_grid_points : int (default=10)
        Number of grid points for sampling n_samples / n_features between (0,1)

    verbose : bool (default=False)
        Print out progress information.

    n_jobs: int, optional
        number of jobs to run in parallel (default 1).

    Methods
    ----------
    show() : Plot the results.

    Attributes
    ----------
    grid_ : array of size (n_grid_points, )
        Array of n_samples / n_features ratios.

    alphas_ : array of size (n_alpha_grid_points, )
        Array of alphas used to generate test graphs 
        (see .statistical_power.new_graph)

    ks_ : array of size (n_alpha_grid_points, )
        The sparsity of each test graph.

    results_ : matrix of size (n_alpha_grid_points, n_grid_points)
        The statisical power, P(exact support recovery) for each alpha and 
        n_samples / n_features grid point.
    """
    def __init__(self, model_selection_estimator=None, n_features=50, 
                trial_estimator=None, n_trials=100, n_grid_points=10, adj_type='erdos-renyi',
                verbose=False, penalty_='lam_', penalty='lam', n_jobs=1):
        self.model_selection_estimator = model_selection_estimator  
        self.trial_estimator = trial_estimator
        self.n_features = n_features
        self.n_grid_points = n_grid_points
        self.adj_type = adj_type
        self.n_trials = n_trials
        self.verbose = verbose
        self.penalty_ = penalty_ # class name for model selected penalty
        self.penalty = penalty # class name for setting penalty
        self.n_jobs = n_jobs

        self.is_fitted = False
        self.results_ = None
        self.alphas_ = None
        self.ks_ = None
        self.grid_ = None
 
    def fit(self, X=None, y=None):
        n_alpha_grid_points = 4

        self.results_ = np.zeros((n_alpha_grid_points, self.n_grid_points))
        self.grid_ = np.logspace(0, np.log10(200), self.n_grid_points)
        if self.adj_type=='erdos-renyi':
            self.alphas_ = np.logspace(-2.3,np.log10(.025), n_alpha_grid_points)[::1]
        else:
            self.alphas_ = np.logspace(np.log(.1),np.log10(.3), n_alpha_grid_points)[::1]

        self.ks_ = []

        for aidx, alpha in enumerate(self.alphas_):
            if self.verbose:
                print 'at alpha {} ({}/{})'.format(
                    alpha,
                    aidx,
                    n_alpha_grid_points,
                )
            
            # draw a new fixed graph for alpha
            cov, prec, adj = new_graph(self.n_features, alpha, adj_type=self.adj_type,random_sign=False,seed=1)    
            n_nonzero_prec = np.count_nonzero(np.triu(adj,1).flat)
            self.ks_.append(n_nonzero_prec)
            mcmc_prng = np.random.RandomState(2)
            if self.verbose:
                print '   Graph has {} nonzero entries'.format(n_nonzero_prec)

            for sidx, sample_grid in enumerate(self.grid_):
                n_samples = int(sample_grid * self.n_features)
                # Debugging
                # print alpha, n_samples
                
                # model selection (once)
                X = mvn(n_samples, self.n_features, cov,random_state=mcmc_prng)
                ms_estimator = clone(self.model_selection_estimator)
                ms_estimator.fit(X)                
                lam = getattr(ms_estimator, self.penalty_)
                
                if self.verbose:
                    display_lam = lam
                    if isinstance(lam, np.ndarray):
                        display_lam = np.linalg.norm(lam)
                    print '   ({}/{}), n_samples = {}, selected lambda = {}'.format(
                            sidx,
                            self.n_grid_points,
                            n_samples,
                            display_lam)

                # setup default trial estimator
                if self.trial_estimator is None:
                    trial_estimator = QuicGraphLasso(lam=lam,
                                                     mode='default',
                                                     init_method='corrcoef')
                elif self.trial_estimator == 'Adaptive':
                    trial_estimator = AdaptiveGraphLasso(estimator = QuicGraphLasso(lam=lam,mode='default',init_method='corrcoef'), 
                                                         method='inverse_squared')
                else:
                    trial_estimator = self.trial_estimator

                # patch trial estimator with this lambda
                if self.trial_estimator == 'Adaptive':
                    trial_estimator.estimator_.set_params(**{
                        self.penalty: lam, 
                    })
                else:
                    trial_estimator.set_params(**{
                        self.penalty: lam, 
                    })
                    

                # estimate statistical power
                exact_support_counts = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=False,
                    backend='threading',
                    #max_nbytes=None,
                    #batch_size=1,
                )(
                    delayed(_sp_trial)(
                        trial_estimator, n_samples, self.n_features, cov, adj, mcmc_prng
                    )
                    for nn in range(self.n_trials))

                self.results_[aidx, sidx] = 1. * np.sum(exact_support_counts) / self.n_trials

            if self.verbose:
                print 'Results at this row: {}'.format(self.results_[aidx, :])

        self.is_fitted = True
        return self

    def show(self):
        if not self.is_fitted:
            print 'Not fitted.'
            return

        _plot_spower(self.results_, self.grid_, self.ks_)
