from medil.gues import InputData
import numpy as np


def test_chain_reduction():
    obj = InputData(np.empty((2, 2)))
    cpdag = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    correct_cpdag = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0]])
    correct_ccs = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ]
    )

    r_cpdag, r_ccs = obj.chain_reduction(cpdag, np.eye(len(cpdag)))
    assert (correct_cpdag == r_cpdag).all()
    assert (correct_ccs == r_ccs).all()


def test_topological_sort():
    obj = InputData(np.empty((2, 2)))

    inv_order = np.array([4, 0, 2, 1, 3])
    dag = np.triu(np.ones((5, 5)), 1)[inv_order][:, inv_order]

    correct_idx = np.argsort(inv_order)
    assert (correct_idx == obj.topological_sort(dag)).all()


def test_get_min_ants():
    obj = InputData(np.empty((7, 7)))
    obj.cpdag = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    correct_min_ants = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 0],
        ]
    )

    assert (obj.get_min_ants() == correct_min_ants).all()


def test_rmable_edges():
    obj = InputData(np.empty((7, 7)))
    obj.cpdag = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    undir_rmable, dir_rmable = obj.rmable_edges()
    assert (undir_rmable == np.array([[1, 2], [1, 3], [2, 3]])).all()
    assert (dir_rmable == np.array([[5, 6]])).all()
