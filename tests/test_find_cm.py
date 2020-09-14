import numpy as np
from medil.ecc_algorithms import find_clique_min_cover as find_cm
from medil.examples import examples


# Here are some integration tests.
# def test_find_cm_on_trivial_edgeless_graph():
#     # not so important at the moment
#     graph = np.array([[]], int)
#     cover = find_cm(graph, True)

#     graph = np.zeros((5, 5), int)
#     cover = find_cm(graph, True)


def test_find_cm_on_3_cycle():
    cycle_3 = np.ones((3, 3), dtype=int)
    cover = find_cm(cycle_3)
    assert cover.shape == (1, 3)
    assert ~np.any(cover - [1, 1, 1], axis=1)


def test_reduction_rule_1_on_3cycle_plus_isolated():
    graph = np.zeros((4, 4), dtype=int)
    graph[1:4, 1:4] = 1
    graph[0, 0] = 1

    cover = find_cm(graph)

    assert cover.shape == (1, 4)
    assert ~np.any(cover - [0, 1, 1, 1], axis=1)


def test_find_cm_on_examples():
    for example in examples:
        cover = find_cm(example.UDG)
        correct_cover = example.MCM
        assert cover.shape == correct_cover.shape
        for clique in correct_cover:
            assert (~np.any(cover - clique, axis=1)).any()
