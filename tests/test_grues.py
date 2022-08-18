import numpy as np
import medil.grues


def examp_init():
    return np.array(
        [
            [0, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
        ],
        bool,
    )


def examp_cpdag():
    return np.array(
        [
            [0, 1, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
        ],
        bool,
    )


def examp_dag_reduction():
    return np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        bool,
    )


def examp_chain_comps():
    return np.array(
        [
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 1],
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
    src_1, src_2 = obj.pick_source_nodes("merge")

    assert (obj.dag_reduction[:, [src_1, src_2]] == 0).all()
    assert obj.dag_reduction[src_1] @ obj.dag_reduction[src_2]

    obj.chain_comps = examp_chain_comps()
    source = obj.pick_source_nodes("split")

    assert obj.dag_reduction[:, source].sum() == 0
    assert obj.chain_comps[source].sum() > 1

    src_1, src_2 = obj.pick_source_nodes("fiber")
    ch_1_mask, ch_2_mask = obj.dag_reduction[(src_1, src_2), :]

    assert ch_1_mask.sum() and ch_2_mask.sum()
    assert (obj.chain_comps[src_1].sum() > 1) or (ch_1_mask @ ~ch_2_mask)


def test_perform_merge():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = examp_chain_comps()
    obj.dag_reduction = examp_dag_reduction()
    obj.perform_merge(4, 1)

    correct_dag_reduction = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        bool,
    )
    correct_chain_comps = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 1],
        ],
        bool,
    )

    assert (obj.dag_reduction == correct_dag_reduction).all()
    assert (obj.chain_comps == correct_chain_comps).all()


def test_consider_split():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = examp_chain_comps()
    obj.dag_reduction = examp_dag_reduction()
    v, w, source = obj.consider_split()

    chain_comp_mask = examp_chain_comps()[source]
    assert chain_comp_mask[v]
    assert chain_comp_mask[w]
    assert examp_dag_reduction()[:, source].sum() == 0


def test_perform_split():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = examp_chain_comps()
    obj.dag_reduction = examp_dag_reduction()
    v, w, source = 2, 4, 2
    obj.perform_split(v, w, source)

    correct_dag_reduction = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
        ],
        bool,
    )
    correct_chain_comps = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
        ],
        bool,
    )

    assert (obj.dag_reduction == correct_dag_reduction).all()
    assert (obj.chain_comps == correct_chain_comps).all()


def test_consider_fiber():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = ex_cc = examp_chain_comps()
    obj.dag_reduction = ex_d_r = examp_dag_reduction()
    _, src_1, src_2, t, v = obj.consider_fiber()

    assert (t == src_1 and ex_cc[t].sum() > 1) or (
        ex_d_r[src_2, t] and ~ex_d_r[src_1, t]
    )
    assert ex_cc[t, v]


def test_perform_fiber():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = ex_cc = examp_chain_comps()
    obj.dag_reduction = ex_d_r = examp_dag_reduction()
    # within, src_1, src_2, t, v =
