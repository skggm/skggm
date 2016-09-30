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
    ModelAverage
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



def exact_support(prec, prec_hat):
    # Q: why do we need something like this?, and why must eps be so big?
    # Q: can we automatically determine what this threshold should be?    
    eps = .2
    #eps = np.finfo(prec_hat.dtype).eps # too small
    prec_hat[np.abs(prec_hat) <= eps] = 0.0
    
    not_empty = np.count_nonzero(np.triu(prec,1))>0
    
    return not_empty & np.array_equal(
            np.nonzero(np.triu(prec,1).flat)[0],
            np.nonzero(np.triu(prec_hat,1).flat)[0])

def count_support_diff(m, m_hat):
    
    n_features, _ = m.shape

    m_no_diag = m.copy()
    m_no_diag[np.diag_indices(n_features)] = 0
    m_hat_no_diag = m_hat.copy()
    m_hat_no_diag[np.diag_indices(n_features)] = 0

    m_nnz = len(np.nonzero(m_no_diag.flat)[0])
    m_hat_nnz = len(np.nonzero(m_hat_no_diag.flat)[0])

    nnz_intersect = len(np.intersect1d(np.nonzero(m_no_diag.flat)[0],
                                       np.nonzero(m_hat_no_diag.flat)[0]))
    return m_nnz + m_hat_nnz - (2 * nnz_intersect)





def approx_support(prec, prec_hat, prob=.01):
    """
    Returns True if model selection error is less than or equal to prob%
    """        
    #eps = .2
    #prec_hat[np.abs(prec_hat) <= eps] = 0.0

    # Why does np.nonzero/np.flatnonzero create so much problems? 
    A = np.flatnonzero(np.triu(prec,1))
    B = np.flatnonzero(np.triu(prec_hat,1))
    ud = np.flatnonzero(np.triu(np.ones(prec.shape),1))
    notA = np.setdiff1d(ud,A)

    B_in_A_bool = np.in1d(B,A) # true positives
    B_notin_A_bool = np.in1d(B,notA) # false positives
    #print np.sum(B_in_A_bool), np.shape(A)[0]
    #print np.sum(B_notin_A_bool), np.shape(notA)[0]
    
    if np.shape(A)[0]:
        tpr = float(np.sum(B_in_A_bool))/len(A)
        tnr = 1.0-tpr
    else:
        tpr = 0.0
        tnr = 0.0        
    if np.shape(notA)[0]:
        fpr = float(np.sum(B_notin_A_bool))/len(notA)
    else:
        fpr = 0.0
        
    #print tnr,fpr
    
    return np.less_equal(tnr+fpr,prob), tpr, fpr



def _support_diff(m, m_hat):
    '''Count the number of different elements in the support in one triangle,
    not including the diagonal. 
    '''
    n_features, _ = m.shape

    m_no_diag = m.copy()
    m_no_diag[np.diag_indices(n_features)] = 0
    m_hat_no_diag = m_hat.copy()
    m_hat_no_diag[np.diag_indices(n_features)] = 0

    m_nnz = len(np.nonzero(m_no_diag.flat)[0])
    m_hat_nnz = len(np.nonzero(m_hat_no_diag.flat)[0])

    nnz_intersect = len(np.intersect1d(np.nonzero(m_no_diag.flat)[0],
                                       np.nonzero(m_hat_no_diag.flat)[0]))
    return (m_nnz + m_hat_nnz - (2 * nnz_intersect)) / 2.0


def _false_support(m, m_hat):
    '''Count the number of false positive and false negatives supports in 
    m_hat in one triangle, not including the diagonal.
    '''
    n_features, _ = m.shape

    m_no_diag = m.copy()
    m_no_diag[np.diag_indices(n_features)] = 0
    m_hat_no_diag = m_hat.copy()
    m_hat_no_diag[np.diag_indices(n_features)] = 0

    m_nnz = len(np.nonzero(m_no_diag.flat)[0])
    m_hat_nnz = len(np.nonzero(m_hat_no_diag.flat)[0])

    nnz_intersect = len(np.intersect1d(np.nonzero(m_no_diag.flat)[0],
                                       np.nonzero(m_hat_no_diag.flat)[0]))

    false_positives = (m_hat_nnz - nnz_intersect) / 2.0
    false_negatives = (m_nnz - nnz_intersect) / 2.0
    return false_positives, false_negatives


# MCMC Statistical Power
def plot_spower(results, grid, ks):
    plt.figure()
    plt.plot(grid, results.T, lw=2)
    plt.xlabel('n/p (n_samples / n_features)')
    plt.ylabel('P(exact recovery)')
    legend_text = []
    for ks in ks:
        legend_text.append('sparsity={}'.format(ks))
    plt.legend(legend_text)
    plt.show()


def plot_avg_error(results, grid, ks):
    plt.figure()
    plt.plot(grid, results.T, lw=2)
    plt.xlabel('n/p (n_samples / n_features)')
    plt.ylabel('Average error')
    legend_text = []
    for ks in ks:
        legend_text.append('sparsity={}'.format(ks))
    plt.legend(legend_text)
    plt.show()

def ae_trial(trial_estimator, n_samples, n_features, cov, adj, random_state, X = None):
    
    if X is None:
        X = mvn(n_samples, n_features, cov,random_state=random_state)
        new_estimator = clone(trial_estimator)
        new_estimator.fit(X)
        new_precision = new_estimator.precision_        
    else: 
        new_precision = trial_estimator.precision_

    error_fro = np.linalg.norm(np.triu(adj,1) - np.triu(new_precision,1), ord='fro')
    error_supp = _support_diff(adj, new_precision)
    error_fp, error_fn = _false_support(adj, new_precision)
    #error_inf = np.linalg.norm(np.triu(adj,1) - np.triu(new_precision,1), ord=inf) 
    frob_support = np.equal(np.triu(adj,1),0)
    frob_support[np.nonzero(np.tril(np.ones(np.shape(adj)),0))] = False
    error_supp_fro = np.linalg.norm(np.dot(frob_support,np.triu(adj,1)) - np.dot(frob_support,np.triu(new_precision,1)), ord='fro')

    return error_fro, error_supp, error_fp, error_fn, error_supp_fro


def sp_trial(trial_estimator, n_samples, n_features, cov, adj, random_state):
    X = mvn(n_samples, n_features, cov,random_state=random_state)
    new_estimator = clone(trial_estimator)
    new_estimator.fit(X)
    return exact_support(adj, new_estimator.precision_)


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
        
    adj_type: string (default='erdos-renyi')
        Name of the type of structure used to create adjacency matrix
        
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
                    delayed(sp_trial)(
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

        plot_spower(self.results_, self.grid_, self.ks_)

class AverageError(object):
    """Compute the average error of a model selector for
    different values of alpha over grid of n_samples / n_features.

    Use this to compare model selection methods.

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

    penalty_ : string (default='lam_')
        Name of the selected best penalty in estimator
        e.g., 'lam_' for QuicGraphLassoCV, QuicGraphLassoEBIC,
              'alpha_' for GraphLassoCV

    n_grid_points : int (default=10)
        Number of grid points for sampling n_samples / n_features between (0,1)

    adj_type: string (default='erdos-renyi')
        Name of the type of structure used to create adjacency matrix
    
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
        (see .statistical_power._new_graph)

    ks_ : array of size (n_alpha_grid_points, )
        The sparsity of each test graph.

    error_fro_ : matrix of size (n_alpha_grid_points, n_grid_points)
        The average Frobenius error for each alpha and 
        n_samples / n_features grid point.

    error_supp_ : matrix of size (n_alpha_grid_points, n_grid_points)
        The average support difference for each alpha and 
        n_samples / n_features grid point.

    error_fp_ : matrix of size (n_alpha_grid_points, n_grid_points)
        The average false positive difference for each alpha and 
        n_samples / n_features grid point.

    error_fn_ : matrix of size (n_alpha_grid_points, n_grid_points)
        The average false negative difference for each alpha and 
        n_samples / n_features grid point.
    """
    def __init__(self, model_selection_estimator=None, n_features=50, 
                n_trials=100, n_grid_points=10, adj_type='erdos-renyi', verbose=False, penalty_='lam_',
                n_jobs=1):
        self.model_selection_estimator = model_selection_estimator  
        self.n_features = n_features
        self.n_grid_points = n_grid_points
        self.adj_type = adj_type
        self.n_trials = n_trials
        self.verbose = verbose
        self.penalty_ = penalty_ # class name for model selected penalty
        self.n_jobs = n_jobs
    
        self.is_fitted = False
        self.error_fro_ = None
        self.error_supp_ = None
        self.error_fp_ = None
        self.error_fn_ = None
        self.alphas_ = None
        self.ks_ = None
        self.grid_ = None
 
    def fit(self, X=None, y=None):
        n_alpha_grid_points = 4

        self.error_fro_ = np.zeros((n_alpha_grid_points, self.n_grid_points))
        self.error_supp_ = np.zeros((n_alpha_grid_points, self.n_grid_points))
        self.error_fp_ = np.zeros((n_alpha_grid_points, self.n_grid_points))
        self.error_fn_ = np.zeros((n_alpha_grid_points, self.n_grid_points))

        self.grid_ = np.linspace(5, 200, self.n_grid_points)
        #self.grid_ = np.logspace(np.log10(2), np.log10(200), self.n_grid_points)
        if self.adj_type=='erdos-renyi':
            self.alphas_ = np.logspace(-2.3,np.log10(.025), n_alpha_grid_points)[::1]
            #self.alphas_ = np.linspace(0.95, 0.99, n_alpha_grid_points)[::-1]
        else:
            self.alphas_ = np.logspace(np.log(.15),np.log10(.4), n_alpha_grid_points)[::1]
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
            # cov, prec = _new_graph(self.n_features, alpha)
            # n_nonzero_prec = np.count_nonzero(prec.flat)
            # self.ks_.append(n_nonzero_prec)
            
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
                trial_estimator = QuicGraphLasso(lam=lam,
                                                 mode='default',
                                                 init_method='corrcoef')

                # estimate statistical power
                errors = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=False,
                    backend='threading',
                    #max_nbytes=None,
                    #batch_size=1,
                )(
                    delayed(ae_trial)(
                        trial_estimator, n_samples, self.n_features, cov, adj, random_state=mcmc_prng
                    )
                    for nn in range(self.n_trials))

                error_fro, error_supp, error_fp, error_fn, _ = zip(*errors)
                self.error_fro_[aidx, sidx] = np.mean(error_fro)
                self.error_supp_[aidx, sidx] = np.mean(error_supp)
                self.error_fp_[aidx, sidx] = np.mean(error_fp)
                self.error_fn_[aidx, sidx] = np.mean(error_fn)

            if self.verbose:
                print 'Results at this row:'
                print '   fro = {}'.format(self.error_fro_[aidx, :])
                print '   supp = {}'.format(self.error_supp_[aidx, :])
                print '   fp = {}'.format(self.error_fp_[aidx, :])
                print '   fn = {}'.format(self.error_fn_[aidx, :])

        self.is_fitted = True
        return self

    def show(self):
        if not self.is_fitted:
            print 'Not fitted.'
            return

        errors_to_plot = [
            ('Frobenius', self.error_fro_),
            ('Support', self.error_supp_),
            ('False Positive', self.error_fp_),
            ('False Negative', self.error_fn_),
        ]
        for name, result in errors_to_plot:
            plot_avg_error(result, self.grid_, self.ks_)
            plt.title(name)        