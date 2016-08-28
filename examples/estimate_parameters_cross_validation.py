import sys
import numpy as np
import tabulate
import time

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
from inverse_covariance.inverse_covariance import _compute_error


'''
Compare inverse covariance estimators and model selection methods.

Derived from example in:    
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


def show_results(covs, precs):
    # plot the covariances
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)
    for i, (name, this_cov) in enumerate(covs):
        vmax = np.abs(this_cov).max()
        plt.subplot(7, 3, i + 1)
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
        ax = plt.subplot(7, 3, i + 1)
        plt.imshow(np.ma.masked_values(this_prec, 0),
                   interpolation='nearest', vmin=-vmax, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        if lam == '':
            plt.title('{}'.format(name))
        else:
            plt.title('{}\n(lam={:.2f})'.format(name, lam))
        ax.set_axis_bgcolor('.7')

    plt.show()


def quic_graph_lasso(X, num_folds, metric):
    '''Run QuicGraphLasso with mode='default' and use standard scikit  
    GridSearchCV to find the best lambda.  

    Primarily demonstrates compatibility with existing scikit tooling. 
    '''
    print 'QuicGraphLasso + GridSearchCV with:'
    print '   metric: {}'.format(metric)
    search_grid = {
      'lam': np.logspace(np.log10(0.01), np.log10(1.0), num=100, endpoint=True),
      'init_method': ['cov', 'corrcoef'],
      'score_metric': [metric], 
    }
    model = GridSearchCV(QuicGraphLasso(),
                         search_grid,
                         cv=num_folds,
                         refit=True)
    model.fit(X)
    bmodel = model.best_estimator_
    print '   len(cv_lams): {}'.format(len(search_grid['lam']))
    print '   cv-lam: {}'.format(model.best_params_['lam'])
    print '   lam_scale_: {}'.format(bmodel.lam_scale_)
    print '   lam_: {}'.format(bmodel.lam_)
    return bmodel.covariance_, bmodel.precision_, bmodel.lam_


def quic_graph_lasso_cv(X, metric):
    '''Run QuicGraphLassoCV on data with metric of choice.

    Compare results with GridSearchCV + quic_graph_lasso.  The number of lambdas
    tested should be much lower with similar final lam_ selected.
    '''
    print 'QuicGraphLassoCV with:'
    print '   metric: {}'.format(metric)
    model = QuicGraphLassoCV(
            cv=2, # cant deal w more folds at small size
            n_refinements=6,
            n_jobs=1,
            init_method='cov',
            score_metric=metric)
    model.fit(X)
    print '   len(cv_lams): {}'.format(len(model.cv_lams_))
    print '   lam_scale_: {}'.format(model.lam_scale_)
    print '   lam_: {}'.format(model.lam_)
    return model.covariance_, model.precision_, model.lam_


def adaptive_graph_lasso(X, model_selector, method):
    '''Run QuicGraphLassoCV or QuicGraphLassoEBIC as a two step adaptive fit
    with method of choice (currently: 'binary', 'inverse', 'inverse_squared').

    Compare the support and values to the model-selection estimator.
    '''
    metric = 'frobenius'
    print 'Adaptive {} with:'.format(model_selector)
    print '   metric: {}'.format(metric)  
    print '   adaptive-method: {}'.format(method)  
    if model_selector == 'QuicGraphLassoCV':
        model = AdaptiveGraphLasso(
                estimator=QuicGraphLassoCV(score_metric=metric),
                method=method,
        )
    
    elif model_selector == 'QuicGraphLassoEBIC':
        model = AdaptiveGraphLasso(
                estimator=QuicGraphLassoEBIC(),
                method=method,
        )
    model.fit(X)
    lam_norm_ = np.linalg.norm(model.estimator_.lam_)
    print '   ||lam_||_2: {}'.format(lam_norm_)
    return model.estimator_.covariance_, model.estimator_.precision_, lam_norm_


def quic_graph_lasso_ebic_manual(X, gamma=0):
    '''Run QuicGraphLasso with mode='path' and gamma; use EBIC criteria for model 
    selection.  

    The EBIC criteria is built into InverseCovarianceEstimator base class 
    so we demonstrate those utilities here.  
    '''
    print 'QuicGraphLasso (manual EBIC) with:'
    print '   mode: path'
    print '   gamma: {}'.format(gamma)
    model = QuicGraphLasso(
        lam=1.0,
        mode='path',
        init_method='cov',
        path=np.logspace(np.log10(0.01), np.log10(1.0), num=100, endpoint=True))
    model.fit(X)
    ebic_index = model.ebic_select(gamma=gamma)
    covariance_ = model.covariance_[ebic_index]
    precision_ = model.precision_[ebic_index]
    lam_ = model.lam_at_index(ebic_index)
    print '   len(path lams): {}'.format(len(model.path))
    print '   lam_scale_: {}'.format(model.lam_scale_)
    print '   lam_: {}'.format(lam_)
    print '   ebic_index: {}'.format(ebic_index)
    return covariance_, precision_, lam_


def quic_graph_lasso_ebic(X, gamma=0):
    '''Run QuicGraphLassoEBIC with gamma.

    QuicGraphLassoEBIC is a convenience class.  Results should be identical to
    those obtained via quic_graph_lasso_ebic_manual.
    '''
    print 'QuicGraphLassoEBIC with:'
    print '   mode: path'
    print '   gamma: {}'.format(gamma)
    model = QuicGraphLassoEBIC(
        lam=1.0,
        init_method='cov',
        gamma=gamma)
    model.fit(X)
    print '   len(path lams): {}'.format(len(model.path))
    print '   lam_scale_: {}'.format(model.lam_scale_)
    print '   lam_: {}'.format(model.lam_)
    return model.covariance_, model.precision_, model.lam_


def empirical(X):
    '''Compute empirical covariance as baseline estimator.
    '''
    print 'Empirical'
    cov = np.dot(X.T, X) / n_samples
    return cov, np.linalg.inv(cov)


def graph_lasso(X, num_folds):
    '''Estimate inverse covariance via scikit-learn GraphLassoCV class.
    '''
    print 'GraphLasso (sklearn)'
    model = GraphLassoCV(cv=num_folds)
    model.fit(X)
    print '   lam_: {}'.format(model.alpha_)
    return model.covariance_, model.precision_, model.alpha_


def sk_ledoit_wolf(X):
    '''Estimate inverse covariance via scikit-learn ledoit_wolf function.
    '''
    lw_cov_, _ = ledoit_wolf(X)
    lw_prec_ = np.linalg.inv(lw_cov_)
    return lw_cov_, lw_prec_


if __name__ == "__main__":
    n_samples = 100
    n_features = 20
    cv_folds = 3

    # make data
    X, true_cov, true_prec = make_data(n_samples, n_features)
    
    plot_covs = [('True', true_cov),
                 ('True', true_cov),
                 ('True', true_cov)]
    plot_precs = [('True', true_prec, ''),
                  ('True', true_prec, ''),
                  ('True', true_prec, '')]
    results = []

    # Empirical
    start_time = time.time()
    cov, prec = empirical(X)
    end_time = time.time()
    ctime = end_time - start_time
    name = 'Empirical'
    plot_covs.append((name, cov))
    plot_precs.append((name, prec, ''))
    error = _compute_error(true_cov, cov, prec)
    results.append([name, error, '', ctime])
    print '   frobenius error: {}'.format(error)
    print ''

    # sklearn LedoitWolf
    start_time = time.time()
    cov, prec = sk_ledoit_wolf(X)
    end_time = time.time()
    ctime = end_time - start_time
    name = 'Ledoit-Wolf (sklearn)'
    plot_covs.append((name, cov))
    plot_precs.append((name, prec, ''))
    error = _compute_error(true_cov, cov, prec)
    results.append([name, error, '', ctime])
    print '   frobenius error: {}'.format(error)
    print ''

    # sklearn GraphLassoCV
    start_time = time.time()
    cov, prec, lam = graph_lasso(X, cv_folds)
    end_time = time.time()
    ctime = end_time - start_time
    name = 'GraphLassoCV (sklearn)'
    plot_covs.append((name, cov))
    plot_precs.append((name, prec, lam))
    error = _compute_error(true_cov, cov, prec)
    results.append([name, error, lam, ctime])
    print '   frobenius error: {}'.format(error)
    print ''
    
    # QuicGraphLasso + GridSearchCV
    params = [
        ('QuicGraphLasso GSCV : ll', 'log_likelihood'),
        ('QuicGraphLasso GSCV : kl', 'kl'),
        ('QuicGraphLasso GSCV : fro', 'frobenius'),
    ]
    for name, metric in params:
        start_time = time.time()
        cov, prec, lam = quic_graph_lasso(X, cv_folds, metric=metric)
        end_time = time.time()
        ctime = end_time - start_time
        plot_covs.append((name, cov))
        plot_precs.append((name, prec, lam))
        error = _compute_error(true_cov, cov, prec)
        results.append([name, error, lam, ctime])
        print '   frobenius error: {}'.format(error)
        print ''

    # QuicGraphLassoCV
    params = [
        ('QuicGraphLassoCV : ll', 'log_likelihood'),
        ('QuicGraphLassoCV : kl', 'kl'),
        ('QuicGraphLassoCV : fro', 'frobenius'),
    ]
    for name, metric in params:
        start_time = time.time()
        cov, prec, lam = quic_graph_lasso_cv(X, metric=metric)
        end_time = time.time()
        ctime = end_time - start_time
        plot_covs.append((name, cov))
        plot_precs.append((name, prec, lam))
        error = _compute_error(true_cov, cov, prec)
        results.append([name, error, lam, ctime])
        print '   frobenius error: {}'.format(error)
        print ''

    # QuicGraphLassoEBIC
    params = [
        ('QuicGraphLassoEBIC : BIC', 0),
        ('QuicGraphLassoEBIC : g=0.01', 0.01),
        ('QuicGraphLassoEBIC : g=0.1', 0.1),
    ]
    for name, gamma in params:
        start_time = time.time()
        # cov, prec, lam = quic_graph_lasso_ebic_manual(X, gamma=gamma)
        cov, prec, lam = quic_graph_lasso_ebic(X, gamma=gamma)
        end_time = time.time()
        ctime = end_time - start_time
        plot_covs.append((name, cov))
        plot_precs.append((name, prec, lam))
        error = _compute_error(true_cov, cov, prec)
        results.append([name, error, lam, ctime])
        print '   error: {}'.format(error)
        print ''

    # Adaptive QuicGraphLassoCV and QuicGraphLassoEBIC
    params = [
        ('Adaptive CV : binary', 'QuicGraphLassoCV', 'binary'),
        ('Adaptive CV : inv', 'QuicGraphLassoCV', 'inverse'),
        ('Adaptive CV : inv**2', 'QuicGraphLassoCV', 'inverse_squared'),
        ('Adaptive BIC : binary', 'QuicGraphLassoEBIC', 'binary'),
        ('Adaptive BIC : inv', 'QuicGraphLassoEBIC', 'inverse'),
        ('Adaptive BIC : inv**2', 'QuicGraphLassoEBIC', 'inverse_squared'),
    ]
    for name, model_selector, method in params:
        start_time = time.time()
        cov, prec, lam = adaptive_graph_lasso(X, model_selector, method)
        end_time = time.time()
        ctime = end_time - start_time
        plot_covs.append((name, cov))
        plot_precs.append((name, prec, ''))
        error = _compute_error(true_cov, cov, prec)
        results.append([name, error, '', ctime])
        print '   frobenius error: {}'.format(error)
        print ''

    # tabulate errors
    print tabulate.tabulate(results,
                            headers=['Estimator', 'Error (Frobenius)', 'Lambda', 'Time'],
                            tablefmt='pipe')
    print ''

    # display results
    show_results(plot_covs, plot_precs)
    raw_input('Press any key to exit...')
