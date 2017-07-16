import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.externals.six import with_metaclass
from sklearn.cross_validation import _PartitionIterator
from sklearn.utils import check_random_state


class _BaseRepeatedKFold(with_metaclass(ABCMeta, _PartitionIterator)):
    """Base class to validate KFoldRepeated approaches"""

    @abstractmethod
    def __init__(self, n, n_folds, n_trials, random_state):
        super(_BaseRepeatedKFold, self).__init__(n)

        if abs(n_folds - int(n_folds)) >= np.finfo('f').eps:
            raise ValueError("n_folds must be an integer")
        self.n_folds = n_folds = int(n_folds)

        if n_folds <= 1:
            raise ValueError(
                "repeated k-fold cross validation requires at least one"
                " train / test split by setting n_folds=2 or more,"
                " got n_folds={0}.".format(n_folds))
        if n_folds > self.n:
            raise ValueError(
                ("Cannot have number of folds n_folds={0} greater"
                 " than the number of samples: {1}.").format(n_folds, n))

        if not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError("n_trials must be int and greater than 0;"
                             " got {0}".format(n_trials))

        self.n_trials = n_trials
        self.random_state = random_state


class RepeatedKFold(_BaseRepeatedKFold):
    """Repeated K-Folds cross validation iterator.

    Provides train/test indices to split data in train test sets. We reshuffle
    the data n_trials times and split dataset into k consecutive folds for each
    trial.

    Each fold is then used as a validation set once while the k - 1 remaining
    fold(s) form the training set.

    The iterator will generate n_folds * n_trials train/test splits.

    Technique outlined in:
        "Cross-validation pitfalls when selecting and assessing
        regression and classification models"
        D. Krstajic, L. Buturovic, D. Leahy, and S. Thomas
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3994246/

    Parameters
    ----------
    n : int
        Total number of elements.

    n_folds : int, default=3
        Number of folds. Must be at least 2.

    n_trials : int, default=3
        Number of random index shuffles.
        n_trials=1 is equivalent to KFold with shuffle=True.

    random_state : None, int or RandomState
        If None, use default numpy RNG for shuffling.

    See also
    --------
    sklearn.cross_validation.KFold
    sklearn.cross_validation.ShuffleSplit
    """
    def __init__(self, n, n_folds=3, n_trials=3, random_state=None):
        super(RepeatedKFold, self).__init__(n, n_folds, n_trials, random_state)
        rng = check_random_state(self.random_state)

        self.idxs = []
        for tt in range(self.n_trials):
            idxs = np.arange(n)
            rng.shuffle(idxs)
            self.idxs.append(idxs)

    def _iter_test_indices(self):
        n = self.n
        n_folds = self.n_folds
        fold_sizes = (n // n_folds) * np.ones(n_folds, dtype=np.int)
        fold_sizes[:n % n_folds] += 1

        for idxs in self.idxs:
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                yield idxs[start:stop]
                current = stop

    def __repr__(self):
        return '%s.%s(n=%i, n_folds=%i, n_trials=%s, random_state=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.n_folds,
            self.n_trials,
            self.random_state,
        )

    def __len__(self):
        return self.n_folds * self.n_trials
