import numpy as np
import medil.grues


examp_init = np.array(
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
examp_cpdag = np.array(
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
examp_dag_reduction = np.array([[0, 1], [2, 1], [3, 1]])
examp_chain_comps = np.array(
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
    obj = medil.grues.InputData(np.empty((1, len(examp_init))))
    obj.init_uec(examp_init)
    obj.get_max_cpdag()

    correct_cpdag = examp_cpdag

    assert (obj.cpdag == correct_cpdag).all()


def test_reduce_max_cpdag():
    obj = medil.grues.InputData(np.empty((1, len(examp_init))))
    obj.cpdag = examp_cpdag
    obj.reduce_max_cpdag()

    correct_dag_reduction = examp_dag_reduction
    correct_chain_comps = examp_chain_comps

    assert (obj.dag_reduction == correct_dag_reduction).all()
    assert (obj.chain_comps == correct_chain_comps).all()
