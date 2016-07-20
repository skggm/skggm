import sys
import pprint
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphLassoCV, ledoit_wolf
import matplotlib.pyplot as plt

sys.path.append('../inverse_covariance')
from inverse_covariance import InverseCovariance 


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
      'lam': np.logspace(np.log10(0.001), np.log10(1.0), num=50, endpoint=True),
      'path': [np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])],
      'mode': ['path'],
      'initialize_method': ['cov'],
      'metric': [metric],
    }

    # search for best parameters
    estimator = GridSearchCV(InverseCovariance(),
                            search_grid,
                            cv=num_folds,
                            refit=True,
                            verbose=1)
    estimator.fit(X)
    ic_estimator = estimator.best_estimator_
    ic_score = ic_estimator.score(X) # must score() to find best lambda index
    ic_path_index = ic_estimator.score_best_path_scale_index_

    print 'Best parameters:'
    pprint.pprint(estimator.best_params_)
    print 'Best score: {}'.format(ic_score)
    print 'Best lambda path scale {} (index= {}), lam = {}'.format(
        ic_estimator.path[ic_path_index],
        ic_path_index,
        ic_estimator.best_lam)

    # get best covariance from QUIC
    cov = np.reshape(ic_estimator.covariance_[ic_path_index, :],
                    (n_features, n_features))
    prec = np.reshape(ic_estimator.precision_[ic_path_index, :],
                    (n_features, n_features))

    return cov, prec

def estimate_via_quic_ebic(X, gamma=0):
    print '\n-- QUIC EBIC'
    ic_estimator = InverseCovariance(
        lam=1.0,
        mode='path',
        initialize_method='cov',
        path=np.logspace(np.log10(0.001), np.log10(1.0), num=50, endpoint=True))
    ic_estimator.fit(X)

    # ebic model selection
    ic_path_index = ic_estimator.model_select(gamma=gamma)
    print 'Best lambda path scale {} (index= {}), lam = {}'.format(
        ic_estimator.path[ic_path_index],
        ic_path_index,
        ic_estimator.best_lam)

    cov = np.reshape(ic_estimator.covariance_[ic_path_index, :],
                    (n_features, n_features))
    prec = np.reshape(ic_estimator.precision_[ic_path_index, :],
                    (n_features, n_features))

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
        vmax = this_cov.max()
        plt.subplot(3, 4, i + 1)
        plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.title('%s covariance' % name)

    plt.show()

    # plot the precisions
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)
    for i, (name, this_prec) in enumerate(precs):
        vmax = this_prec.max()
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(np.ma.masked_equal(this_prec, 0),
                   interpolation='nearest', vmin=-vmax, vmax=vmax,
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
    cv_folds = 2

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
            ('Quic (ll)', quic_ll_cov),
            ('Quic (kl)', quic_kl_cov),
            ('Quic (fro)', quic_fro_cov),
            ('Quic (bic)', quic_bic_cov),
            ('Quic (ebic gamma = 0.1)', quic_ebic_cov),
            ('GraphLasso', gl_cov),
            ('Ledoit-Wolf', lw_cov)]
    precs = [('True', prec),
            ('Empirical', emp_prec),
            ('Quic (ll)', quic_ll_prec),
            ('Quic (kl)', quic_kl_prec),
            ('Quic (fro)', quic_fro_prec),
            ('Quic (bic)', quic_bic_prec),
            ('Quic (ebic gamma = 0.1)', quic_ebic_prec),
            ('GraphLasso', gl_prec),
            ('Ledoit-Wolf', lw_prec)]
    show_results(covs, precs)

  