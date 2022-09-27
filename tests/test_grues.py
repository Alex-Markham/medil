# pipenv run pytest --cov-report html --cov-report term --cov=medil.grues -v tests/test_grues.py
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

    correct_chain_comps = examp_chain_comps()
    correct_dag_reduction = examp_dag_reduction()

    assert (obj.dag_reduction == correct_dag_reduction).all()
    assert (obj.chain_comps == correct_chain_comps).all()


def test_pick_source_nodes():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.q = {key: 1 for key in ["split", "within", "out_del", "merge"]}
    obj.dag_reduction = examp_dag_reduction()
    obj.chain_comps = examp_chain_comps()

    source = obj.pick_source_nodes("split")

    assert obj.dag_reduction[:, source].sum() == 0
    assert obj.chain_comps[source].sum() > 1

    src_1, src_2 = obj.pick_source_nodes("within")
    ch_1_mask, ch_2_mask = obj.dag_reduction[(src_1, src_2), :]

    assert src_1 != src_2
    assert ch_1_mask.sum() and ch_2_mask.sum()
    assert (obj.chain_comps[src_1].sum() > 1) or (ch_1_mask @ ~ch_2_mask)

    source, t = obj.pick_source_nodes("out_del")

    assert obj.dag_reduction[source, t].all()
    assert obj.dag_reduction[:, source].sum() == 0

    obj.dag_reduction[[2, 3], 0] = True
    src_1, src_2 = obj.pick_source_nodes("merge")

    assert src_1 != src_2
    assert (obj.dag_reduction[:, [src_1, src_2]] == 0).all()
    assert (obj.chain_comps[[src_1, src_2]].sum(1) == 1).all()
    assert (obj.dag_reduction[src_1] == obj.dag_reduction[src_2]).all()


def test_perform_merge():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        bool,
    )
    obj.dag_reduction = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 0],
        ],
        bool,
    )
    obj.perform_merge(5, 6)

    correct_chain_comps = examp_chain_comps()
    correct_dag_reduction = examp_dag_reduction()

    assert (obj.dag_reduction == correct_dag_reduction).all()
    assert (obj.chain_comps == correct_chain_comps).all()


def test_consider_split():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.q = {"split": 1}
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
    v, w, source = 6, 0, 4
    obj.perform_split(v, w, source)

    correct_chain_comps = np.array(
        [
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        bool,
    )
    correct_dag_reduction = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 0],
        ],
        bool,
    )

    assert (obj.chain_comps == correct_chain_comps).all()
    assert (obj.dag_reduction == correct_dag_reduction).all()


def test_consider_algebraic():
    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.q = {key: 1 for key in ["within", "out_del", "out_add"]}
    obj.chain_comps = examp_chain_comps()
    obj.dag_reduction = examp_dag_reduction()

    src_1, _, t, v, T_mask = obj.consider_algebraic("out_del")
    assert obj.chain_comps[t, v]
    assert T_mask[src_1] == False
    assert obj.dag_reduction[T_mask, t].all()
    assert (obj.dag_reduction[:, T_mask] == False).all()

    src_1, src_2, t, v, T_mask = obj.consider_algebraic("within")
    ex_cc = examp_chain_comps()
    ex_d_r = examp_dag_reduction()
    assert (t == src_1 and ex_cc[t].sum() > 1) or (
        ex_d_r[src_1, t] and ~ex_d_r[src_2, t]
    )
    assert ex_cc[t, v]
    assert T_mask[src_1] == False
    assert (obj.dag_reduction[:, T_mask] == False).all()
    T_mask[src_2] = False
    assert ~(T_mask.any()) or obj.dag_reduction[T_mask, t].all()

    src_1, src_2, t, v, T_mask = obj.consider_algebraic("out_add")
    assert T_mask[src_2]
    assert (obj.dag_reduction[:, T_mask] == False).all()
    T_mask[src_2] = False
    assert t == src_1 or obj.dag_reduction[T_mask, t].all()


def test_perform_algebraic_add():
    src_1, src_2, t, v = 2, 4, 3, 5
    T_mask = np.array([0, 1, 1, 0, 1], bool)

    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = examp_chain_comps()
    obj.dag_reduction = examp_dag_reduction()
    obj.perform_algebraic(src_1, src_2, t, v, T_mask)

    correct_chain_comps = examp_chain_comps()[[0, 1, 2, 4, 3]]
    correct_dag_reduction = np.array(
        [
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        bool,
    )

    assert (obj.chain_comps == correct_chain_comps).all()
    assert (obj.dag_reduction == correct_dag_reduction).all()


def test_perform_algebraic_del():
    src_1, src_2, t, v = 4, None, 3, 5
    T_mask = np.array([0, 1, 1, 0, 0], bool)

    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = examp_chain_comps()
    obj.dag_reduction = np.array(
        [
            [0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0],
        ],
        bool,
    )
    obj.perform_algebraic(src_1, src_2, t, v, T_mask)

    order = np.array([0, 1, 2, 4, 3])
    correct_chain_comps = examp_chain_comps()[order]
    correct_dag_reduction = examp_dag_reduction()[np.ix_(order, order)]

    assert (obj.chain_comps == correct_chain_comps).all()
    assert (obj.dag_reduction == correct_dag_reduction).all()


def test_perform_algebraic_within():
    src_1, src_2, t, v = 2, 4, 3, 5
    T_mask = np.array([0, 1, 0, 0, 1], bool)

    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = examp_chain_comps()
    obj.dag_reduction = examp_dag_reduction()
    obj.perform_algebraic(src_1, src_2, t, v, T_mask)

    correct_chain_comps = np.array(
        [
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 1],
        ],
        bool,
    )
    correct_dag_reduction = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        bool,
    )

    assert (obj.chain_comps == correct_chain_comps).all()
    assert (obj.dag_reduction == correct_dag_reduction).all()


def test_perform_algebraic_inv_with():
    src_1, src_2, t, v = 3, 2, 0, 5
    T_mask = np.array([0, 1, 1, 0], bool)

    obj = medil.grues.InputData(np.empty((1, len(examp_init()))))
    obj.chain_comps = np.array(
        [
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 1],
        ],
        bool,
    )
    obj.dag_reduction = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        bool,
    )
    obj.perform_algebraic(src_1, src_2, t, v, T_mask)

    order = np.array([0, 1, 2, 4, 3])
    correct_chain_comps = examp_chain_comps()[order]
    correct_dag_reduction = examp_dag_reduction()[np.ix_(order, order)]

    assert (obj.chain_comps == correct_chain_comps).all()
    assert (obj.dag_reduction == correct_dag_reduction).all()


# integration test
def test_explore():
    samples = np.random.random_sample((50, 3))
    obj = medil.grues.InputData(samples)
    obj.debug = obj.explore = True
    obj.mcmc(max_moves=100)
    assert len(np.unique(obj.visited)) == 8
