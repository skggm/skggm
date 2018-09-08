from sklearn.utils.testing import assert_raises
from sklearn.tests.test_cross_validation import check_cv_coverage
from inverse_covariance import RepeatedKFold


def test_repeated_kfold_coverage():
    n_samples = 300
    n_folds = 3
    n_trials = 3
    kf = RepeatedKFold(n_samples, n_folds, n_trials)
    check_cv_coverage(kf, expected_n_iter=n_folds * n_trials, n_samples=n_samples)

    n_samples = 17
    n_folds = 3
    n_trials = 5
    kf = RepeatedKFold(n_samples, n_folds, n_trials)
    check_cv_coverage(kf, expected_n_iter=n_folds * n_trials, n_samples=n_samples)


def test_repeated_kfold_values():
    # Check that errors are raised if there is not enough samples
    assert_raises(ValueError, RepeatedKFold, 3, 4)

    # Error when number of folds is <= 1
    assert_raises(ValueError, RepeatedKFold, 2, 0)
    assert_raises(ValueError, RepeatedKFold, 2, 1)

    # When n is not integer:
    assert_raises(ValueError, RepeatedKFold, 2.5, 2)

    # When n_folds is not integer:
    assert_raises(ValueError, RepeatedKFold, 5, 1.5)

    # When n_trials is not integer:
    assert_raises(ValueError, RepeatedKFold, 5, 3, 1.5)
