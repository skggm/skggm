import sys
import numpy as np

from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV

from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.metrics import make_scorer

sys.path.append('..')
from inverse_covariance import (
    QuicGraphLasso,
)


def make_data(n_samples, n_features):
    prng = np.random.RandomState(1)
    prec = make_sparse_spd_matrix(
        n_features,
        alpha=.98,
        smallest_coef=.4,
        largest_coef=.7,
        random_state=prng
    )
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

def quic_graph_lasso_gridsearch(X, num_folds, metric):
    '''Run QuicGraphLasso with mode='default' and use standard scikit
    GridSearchCV to find the best lambda.

    Primarily demonstrates compatibility with existing scikit tooling.
    '''
    print('QuicGraphLasso + GridSearchCV with:')
    print('   metric: {}'.format(metric))
    search_grid = {
      'lam': np.logspace(
            np.log10(0.01), np.log10(1.0), num=100, endpoint=True
        )
    }
    model = GridSearchCV(QuicGraphLasso(init_method='corrcoef',score_metric=metric),
                         search_grid,
                         cv=num_folds,
                         refit=True)
    model.fit(X)
    bmodel = model.best_estimator_
    print('   len(cv_lams): {}'.format(len(search_grid['lam'])))
    print('   cv-lam: {}'.format(model.best_params_['lam']))
    print('   lam_scale_: {}'.format(bmodel.lam_scale_))
    print('   lam_: {}'.format(bmodel.lam_))
    return bmodel.covariance_, bmodel.precision_, bmodel.lam_

def quic_graph_lasso_skopt(X, num_folds, metric):
    '''Run QuicGraphLasso with mode='default' and use standard scikit
    GridSearchCV to find the best lambda.

    Primarily demonstrates compatibility with existing scikit tooling.
    '''
    print('QuicGraphLasso + BayesSearchCV with:')
    print('   metric: {}'.format(metric))
    
    model = BayesSearchCV(QuicGraphLasso(init_method='corrcoef', score_metric=metric),
                         cv=num_folds,
                         n_iter=16,
                         refit=True
                         ) # Using n_iter parameter seems to mess up parameter search space
    model.add_spaces('space_1', {
                        'lam': Real(1e-02,1e+1, prior='log-uniform')
                        }
                    )

    model.fit(X)
    score = model.score(X)
    print('   Final score: {}'.format(score))
    bmodel = model.best_estimator_
    print('   cv-lam: {}'.format(model.best_params_['lam']))
    print('   lam_scale_: {}'.format(bmodel.lam_scale_))
    print('   lam_: {}'.format(bmodel.lam_))

    for i in range(2):
        model.step(X,None,'space_1')
        # save the model or use custom stopping criterion here
        # model is updated after every step
        # ...
        score = model.score(X)
        print('   step: {},score: {}'.format(16+i+1, score))
        bmodel = model.best_estimator_
        print('   cv-lam: {}'.format(model.best_params_['lam']))
        print('   lam_scale_: {}'.format(bmodel.lam_scale_))
        print('   lam_: {}'.format(bmodel.lam_))

    return bmodel.covariance_, bmodel.precision_, bmodel.lam_


n_samples = 150;
n_features = 10
X, true_cov, true_prec = make_data(n_samples,n_features)

quic_graph_lasso_gridsearch(X,2,'kl')
quic_graph_lasso_skopt(X,2,'kl')

