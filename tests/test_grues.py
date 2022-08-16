import numpy as np
import medil.grues


def examp_init():
    return np.array(
        [
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 1],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
        ],
        bool,
    )


def examp_cpdag():
    return np.array(
        [
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
        ],
        bool,
    )


def examp_dag_reduction():
    return np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        bool,
    )


def examp_chain_comps():
    return np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
        ],
        bool,
    )


# unit tests
def test_get_max_cpdag():
    init = examp_init()
    obj = medil.grues.InputData(np.empty((1, len(init))))
    obj.init_uec(init)
    obj.get_max_cpdag()

    correct_cpdag = examp_cpdag()

    assert (obj.cpdag == correct_cpdag).all()


def test_reduce_max_cpdag():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.cpdag = examp_cpdag()
    obj.reduce_max_cpdag()

    correct_dag_reduction = examp_dag_reduction()
    correct_chain_comps = examp_chain_comps()

    assert (obj.dag_reduction == correct_dag_reduction).all()
    assert (obj.chain_comps == correct_chain_comps).all()


def test_pick_source_nodes():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.dag_reduction = examp_dag_reduction()
    i, k, j = obj.pick_source_nodes()

    i_idx = np.flatnonzero(obj.dag_reduction[:, 0] == i)
    k_idx = np.flatnonzero(obj.dag_reduction[:, 0] == k)
    assert obj.dag_reduction[i_idx, 1] == j
    assert obj.dag_reduction[k_idx, 1] == j
    assert j not in obj.dag_reduction[:, 0]


def test_perform_merge():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = examp_chain_comps()
    obj.dag_reduction = examp_dag_reduction()
    obj.perform_merge(0, 3)

    correct_dag_reduction = np.array([[2, 1], [3, 1]])
    correct_chain_comps = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 1],
        ]
    )

    assert (obj.dag_reduction == correct_dag_reduction).all()
    assert (obj.chain_comps == correct_chain_comps).all()


def test_consider_split():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = examp_chain_comps()
    obj.dag_reduction = examp_dag_reduction()
    v, w, chosen_cc_idx = obj.consider_split()

    chosen_cc = examp_chain_comps()[chosen_cc_idx]
    assert chosen_cc[v]
    assert chosen_cc[w]
    assert chosen_cc_idx not in examp_dag_reduction()[:, 1]


def test_perform_split():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = examp_chain_comps()
    obj.dag_reduction = examp_dag_reduction()
    v, w, chosen_cc_idx = 2, 4, 2
    obj.perform_split(v, w, chosen_cc_idx)

    correct_dag_reduction = np.array([[0, 1], [2, 1], [3, 1], [4, 1]])
    correct_chain_comps = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 0],
        ],
    )

    assert (obj.dag_reduction == correct_dag_reduction).all()
    assert (obj.chain_comps == correct_chain_comps).all()
