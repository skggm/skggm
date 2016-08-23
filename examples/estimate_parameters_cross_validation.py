import sys
import pprint
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphLassoCV, ledoit_wolf
import matplotlib.pyplot as plt

sys.path.append('..')
from inverse_covariance import (
    QuicGraphLasso,
    QuicGraphLassoCV,
    QuicGraphLassoEBIC,
    AdaptiveGraphLasso,
)


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
      'lam': np.logspace(np.log10(0.01), np.log10(1.0), num=100, endpoint=True),
      'initialize_method': ['cov', 'corrcoef'],
      'score_metric': [metric], # note: score_metrics are not comparable
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

    return ic_estimator.covariance_, ic_estimator.precision_, estimator.best_params_['lam']


def estimate_via_quic_cv(X, num_folds, metric='log_likelihood'):
    print '\n-- QUIC QuicGraphLassoCV'
    model = QuicGraphLassoCV(
            cv=2, # cant deal w more folds at small size
            n_refinements=6,
            n_jobs=1,
            initialize_method='cov',
            score_metric=metric)
    model.fit(X)

    print 'Best parameters:'
    print 'Best lambda: {}'.format(model.lam_)
    print 'CV lams: {}'.format(model.cv_lams_)

    return model.covariance_, model.precision_, model.lam_


def estimate_via_adaptive(X, num_folds):
    print '\n-- AdaptiveGraphLasso'
    #model = AdaptiveGraphLasso(
    #        estimator=QuicGraphLassoCV(score_metric='frobenius'),
    #        method='glasso',
    #)
    model = AdaptiveGraphLasso(
            estimator=QuicGraphLassoEBIC(),
            method='binary',
    )
    model.fit(X)
    return model.estimator_.covariance_, model.estimator_.precision_, -1


def estimate_via_quic_ebic(X, gamma=0):
    print '\n-- QUIC EBIC'
    ic_estimator = QuicGraphLasso(
        lam=1.0,
        mode='path',
        initialize_method='cov',
        path=np.logspace(np.log10(0.01), np.log10(1.0), num=100, endpoint=True))
    ic_estimator.fit(X)

    # ebic model selection
    ebic_index = ic_estimator.ebic_select(gamma=gamma)
    print 'Best lambda path scale {} (index= {}), lam = {}'.format(
        ic_estimator.path[ebic_index],
        ebic_index,
        ic_estimator.lam_at_index(ebic_index))

    cov = ic_estimator.covariance_[ebic_index]
    prec = ic_estimator.precision_[ebic_index]

    return cov, prec, ic_estimator.lam_at_index(ebic_index)


def estimate_via_quic_ebic_convenience(X, gamma=0):
    print '\n-- QUIC EBIC (Convenience)'
    ic_estimator = QuicGraphLassoEBIC(
        lam=1.0,
        initialize_method='cov',
        gamma=gamma)
    ic_estimator.fit(X)

    print 'Best lambda = {}'.format(ic_estimator.lam_)
    return ic_estimator.covariance_, ic_estimator.precision_, ic_estimator.lam_

def estimate_via_empirical(X):
    cov = np.dot(X.T, X) / n_samples
    return cov, np.linalg.inv(cov)


def estimate_via_graph_lasso(X, num_folds):
    model = GraphLassoCV(cv=num_folds)
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
        plt.subplot(4, 4, i + 1)
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
    for i, (name, this_prec, lam) in enumerate(precs):
        vmax = np.abs(this_prec).max()
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(np.ma.masked_values(this_prec, 0),
                   interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        if lam == '':
            plt.title('{}'.format(name))
        else:
            plt.title('{} (lam={:.2f})'.format(name, lam))
        ax.set_axis_bgcolor('.7')

    plt.show()

    raw_input('Press any key to exit...')


if __name__ == "__main__":
    n_samples = 100
    n_features = 20
    cv_folds = 3

    # make data
    X, cov, prec = make_data(n_samples, n_features)
    
    # run estimators
    emp_cov, emp_prec = estimate_via_empirical(X)
    gl_cov, gl_prec = estimate_via_graph_lasso(X, cv_folds)
    lw_cov, lw_prec = estimate_via_ledoit_wolf(X)
    quic_ll_cov, quic_ll_prec, quic_ll_lam = estimate_via_quic(X,
            cv_folds, metric='log_likelihood')
    quic_kl_cov, quic_kl_prec, quic_kl_lam = estimate_via_quic(X,
            cv_folds, metric='kl')
    quic_fro_cov, quic_fro_prec, quic_fro_lam = estimate_via_quic(X,
            cv_folds, metric='frobenius')
    quic_cv_ll_cov, quic_cv_ll_prec, quic_cv_ll_lam = estimate_via_quic_cv(X,
            cv_folds, metric='log_likelihood')
    quic_cv_kl_cov, quic_cv_kl_prec, quic_cv_kl_lam = estimate_via_quic_cv(X,
            cv_folds, metric='kl')
    quic_cv_fro_cov, quic_cv_fro_prec, quic_cv_fro_lam = estimate_via_quic_cv(X,
            cv_folds, metric='frobenius')
    quic_bic_cov, quic_bic_prec, quic_bic_lam = estimate_via_quic_ebic(X, gamma=0)
    quic_ebic_cov, quic_ebic_prec, quic_ebic_lam = estimate_via_quic_ebic_convenience(X, gamma=0.1)
    quic_adaptive_cov, quic_adaptive_prec, quic_adaptive_lam = estimate_via_adaptive(X, cv_folds)

    # Show results
    covs = [('True', cov),
            ('Empirical', emp_cov),
            ('GraphLassoCV', gl_cov),
            ('Ledoit-Wolf', lw_cov),
            ('Quic (cv-ll)', quic_ll_cov),
            ('Quic (cv-kl)', quic_kl_cov),
            ('Quic (cv-fro)', quic_fro_cov),
            ('True', cov),
            ('QuicCV (ll)', quic_cv_ll_cov),
            ('QuicCV (kl)', quic_cv_kl_cov),
            ('QuicCV (fro)', quic_cv_fro_cov),
            ('True', cov),
            ('Quic (bic)', quic_bic_cov),
            ('Quic (ebic gamma = 0.1)', quic_ebic_cov),
            ('Adaptive', quic_adaptive_cov)]
    precs = [('True', prec, ''),
            ('Empirical', emp_prec, ''),
            ('GraphLassoCV', gl_prec, ''),
            ('Ledoit-Wolf', lw_prec, ''),
            ('Quic (cv-ll)', quic_ll_prec, quic_ll_lam),
            ('Quic (cv-kl)', quic_kl_prec, quic_kl_lam),
            ('Quic (cv-fro)', quic_fro_prec, quic_fro_lam),
            ('True', prec, ''),
            ('QuicCV (ll)', quic_cv_ll_prec, quic_cv_ll_lam),
            ('QuicCV (kl)', quic_cv_kl_prec, quic_cv_kl_lam),
            ('QuicCV (fro)', quic_cv_fro_prec, quic_cv_fro_lam),
            ('True', prec, ''),
            ('Quic (bic)', quic_bic_prec, quic_bic_lam),
            ('Quic (ebic gamma = 0.1)', quic_ebic_prec, quic_ebic_lam),
            ('Adaptive', quic_adaptive_prec, quic_adaptive_lam)]
    show_results(covs, precs)

  