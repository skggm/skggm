import sys
import pprint
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphLassoCV, ledoit_wolf
import matplotlib.pyplot as plt

sys.path.append('..')
from inverse_covariance import QuicGraphLasso 


'''
Example of brute-force parameter search with InverseCovariance

Mimic example from:    
http://scikit-learn.org/stable/auto_examples/covariance/plot_sparse_cov.html
'''
plt.ion()

def make_data(n_samples, n_features):
    prng = np.random.RandomState(1)
    prec = make_sparse_spd_matrix(n_features, alpha=.98,
                              smallest_coef=.4,
                              largest_coef=.7,
                              random_state=prng)
    cov = np.linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    return X, cov, prec


def estimate_via_quic(X, num_folds, metric='log_likelihood'):
    print '\n-- QUIC CV'
    search_grid = {
      'lam': np.logspace(np.log10(0.001), np.log10(1.0), num=100, endpoint=True),
      'initialize_method': ['cov', 'corrcoef'],
      'score_metric': [metric],
    }

    # search for best parameters
    estimator = GridSearchCV(QuicGraphLasso(),
                            search_grid,
                            cv=num_folds,
                            refit=True,
                            verbose=1)
    estimator.fit(X)
    ic_estimator = estimator.best_estimator_
    ic_score = ic_estimator.score(X) # must score() to find best lambda index

    print 'Best parameters:'
    pprint.pprint(estimator.best_params_)
    print 'Best score: {}'.format(ic_score)

    # get best covariance from QUIC
    cov = ic_estimator.covariance_
    prec = ic_estimator.precision_

    return cov, prec

def estimate_via_quic_ebic(X, gamma=0):
    print '\n-- QUIC EBIC'
    ic_estimator = QuicGraphLasso(
        lam=1.0,
        mode='path',
        initialize_method='cov',
        path=np.logspace(np.log10(0.001), np.log10(1.0), num=100, endpoint=True))
    ic_estimator.fit(X)

    # ebic model selection
    ebic_index = ic_estimator.ebic_select(gamma=gamma)
    print 'Best lambda path scale {} (index= {}), lam = {}'.format(
        ic_estimator.path[ebic_index],
        ebic_index,
        ic_estimator.lam_select_(ebic_index))

    cov = ic_estimator.covariance_[ebic_index]
    prec = ic_estimator.precision_[ebic_index]

    return cov, prec

def estimate_via_empirical(X):
    cov = np.dot(X.T, X) / n_samples
    return cov, np.linalg.inv(cov)


def estimate_via_graph_lasso(X, num_folds):
    model = GraphLassoCV(cv=num_folds) # default 3
    model.fit(X)
    return model.covariance_, model.precision_


def estimate_via_ledoit_wolf(X):
    lw_cov_, _ = ledoit_wolf(X)
    lw_prec_ = np.linalg.inv(lw_cov_)
    return lw_cov_, lw_prec_


def show_results(covs, precs):
    # plot the covariances
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)
    for i, (name, this_cov) in enumerate(covs):
        vmax = np.abs(this_cov).max()
        plt.subplot(3, 4, i + 1)
        plt.imshow(this_cov, 
                   interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('%s covariance' % name)

    plt.show()

    # plot the precisions
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)
    for i, (name, this_prec) in enumerate(precs):
        vmax = np.abs(this_prec).max()
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(np.abs(this_prec),  #np.ma.masked_values(this_prec, 0)
                   interpolation='nearest', vmin=0, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('%s precision' % name)
        ax.set_axis_bgcolor('.7')

    plt.show()

    raw_input('Press any key to exit...')


if __name__ == "__main__":
    n_samples = 60  
    n_features = 20 
    cv_folds = 3

    # make data
    X, cov, prec = make_data(n_samples, n_features)
    
    # run estimators
    emp_cov, emp_prec = estimate_via_empirical(X)
    gl_cov, gl_prec = estimate_via_graph_lasso(X, cv_folds)
    lw_cov, lw_prec = estimate_via_ledoit_wolf(X)
    quic_ll_cov, quic_ll_prec = estimate_via_quic(X,
            cv_folds, metric='log_likelihood')
    quic_kl_cov, quic_kl_prec = estimate_via_quic(X,
            cv_folds, metric='kl')
    quic_fro_cov, quic_fro_prec = estimate_via_quic(X,
            cv_folds, metric='frobenius')
    quic_bic_cov, quic_bic_prec = estimate_via_quic_ebic(X, gamma=0)
    quic_ebic_cov, quic_ebic_prec = estimate_via_quic_ebic(X, gamma=0.1)

    # Show results
    covs = [('True', cov),
            ('Empirical', emp_cov),
            ('GraphLasso', gl_cov),
            ('Ledoit-Wolf', lw_cov),
            ('Quic (cv-ll)', quic_ll_cov),
            ('Quic (cv-kl)', quic_kl_cov),
            ('Quic (cv-fro)', quic_fro_cov),
            ('Quic (bic)', quic_bic_cov),
            ('Quic (ebic gamma = 0.1)', quic_ebic_cov)]
    precs = [('True', prec),
            ('Empirical', emp_prec),
            ('GraphLasso', gl_prec),
            ('Ledoit-Wolf', lw_prec),
            ('Quic (cv-ll)', quic_ll_prec),
            ('Quic (cv-kl)', quic_kl_prec),
            ('Quic (cv-fro)', quic_fro_prec),
            ('Quic (bic)', quic_bic_prec),
            ('Quic (ebic gamma = 0.1)', quic_ebic_prec)]
    show_results(covs, precs)

  