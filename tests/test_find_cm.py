import numpy as np
from medil.ecc_algorithms import find_clique_min_cover as find_cm
from medil.ecc_algorithms import branch
from medil.ecc_algorithms import max_cliques
from medil.graph import UndirectedDependenceGraph
import medil.examples as ex


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
    graph = np.zeros((4, 4), dtype=int)  # init
    graph[1:4, 1:4] = 1  # add 3cycle
    graph[0, 0] = 1  # add isolated vert

    cover = find_cm(graph)

    assert cover.shape == (1, 4)
    assert ~np.any(cover - [0, 1, 1, 1], axis=1)


def test_find_cm_on_simple_M():
    cover = find_cm(ex.simple_M_UDG())
    correct_cover = ex.simple_M_MCM()
    assert cover.shape == correct_cover.shape
    for clique in correct_cover:
        assert (~np.any(cover - clique, axis=1)).any()


def test_find_cm_on_more_latents():
    cover = find_cm(ex.more_latents_UDG())
    correct_cover = ex.more_latents_MCM()
    assert cover.shape == correct_cover.shape
    for clique in correct_cover:
        assert (~np.any(cover - clique, axis=1)).any()


def test_find_cm_on_triangle():
    cover = find_cm(ex.triangle_UDG())
    correct_cover = ex.triangle_MCM()
    assert cover.shape == correct_cover.shape
    for clique in correct_cover:
        assert (~np.any(cover - clique, axis=1)).any()


def test_find_cm_on_clean_am_cm_diff():
    cover = find_cm(ex.am_cm_diff_UDG())
    correct_cover = ex.am_cm_diff_MCM()
    assert cover.shape == correct_cover.shape
    for clique in correct_cover:
        assert (~np.any(cover - clique, axis=1)).any()
