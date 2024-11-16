import numpy as np
import pytest

from medil.evaluate import nsfd, nshd


def true_mcm():
    return np.array([[True, True, True, False], [False, True, True, True]])


def est_mcm1():
    return np.array([[True, True, True, False], [False, False, True, True]])


def est_mcm2():
    return np.array(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, True, False, True],
            [False, False, True, True],
        ]
    )


def test_nshd():
    with pytest.raises(ValueError):
        nshd([1])
    with pytest.raises(ValueError):
        nshd([1], predicted_biadj=[1], predicted_adj=[1])
    assert nshd(true_mcm(), predicted_biadj=est_mcm1()) == 1 / 6
    assert nshd(true_mcm(), predicted_biadj=est_mcm2()) == 1 / 6
    assert nshd(est_mcm1(), predicted_biadj=est_mcm2()) == 1 / 3


def test_nsfd():
    assert nsfd(true_mcm(), est_mcm1()) == 0.21501600287975522
    assert nsfd(true_mcm(), est_mcm2()) == 0.27774888397299874
    assert nsfd(est_mcm1(), est_mcm2()) == 0.29238200914015666
