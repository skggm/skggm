import numpy as np
import pytest

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import assert_greater

from sklearn import datasets

from ..quic import QUIC

class TestQUIC(object):
    @pytest.mark.parametrize("params_in, results_out", [
        ({}, []),
    ])
    def test_quic(self, params_in, results_out):
        quic = QUIC()
        assert True