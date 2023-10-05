import numpy as np

from medil.evaluate import sfd


def test_sfd():
    true_mcm = np.array([[True, True, True, False], [False, True, True, True]])
    est_mcm1 = np.array([[True, True, True, False], [False, False, True, True]])
    est_mcm2 = np.array(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, True, False, True],
            [False, False, True, True],
        ]
    )

    assert sfd(true_mcm, est_mcm1) == (5, 1)
    assert sfd(true_mcm, est_mcm2) == (10, 1)
    assert sfd(est_mcm1, est_mcm2) == (7, 2)
