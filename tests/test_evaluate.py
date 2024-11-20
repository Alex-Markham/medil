import numpy as np
import pytest

from medil.evaluate import sfd, shd


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
        shd([1])
    with pytest.raises(ValueError):
        shd([1], predicted_biadj=[1], predicted_adj=[1])
    with pytest.raises(ValueError):
        shd(true_mcm(), predicted_biadj=est_mcm1(), to_return="nope")
    assert shd(true_mcm(), predicted_biadj=est_mcm1(), to_return="raw") == 2
    assert shd(true_mcm(), predicted_biadj=est_mcm1(), to_return="normalized") == 2 / 12
    assert shd(true_mcm(), predicted_biadj=est_mcm1(), to_return="both") == (2, 2 / 12)
    assert shd(true_mcm(), predicted_biadj=est_mcm2()) == 2
    assert shd(est_mcm1(), predicted_biadj=est_mcm2()) == 4


def test_nsfd():
    with pytest.raises(ValueError):
        sfd(true_mcm(), est_mcm1(), to_return="nope")
    assert sfd(true_mcm(), est_mcm1(), to_return="raw") == 3
    assert sfd(true_mcm(), est_mcm2(), to_return="raw") == 4
    assert sfd(est_mcm1(), est_mcm2(), to_return="raw") == 5
    assert sfd(true_mcm(), est_mcm1(), to_return="normalized") == 0.21501600287975522
    assert sfd(true_mcm(), est_mcm2(), to_return="normalized") == 0.27774888397299874
    assert sfd(est_mcm1(), est_mcm2(), to_return="normalized") == 0.29238200914015666
    assert sfd(true_mcm(), est_mcm1(), to_return="both") == (3, 0.21501600287975522)
