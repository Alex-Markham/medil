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


def test_find_cm_on_triangle():
    cover = find_cm(ex.triangle_UDG())
    assert cover.shape == (3, 6)

    correct_cover = ex.triangle_MCM()

    assert (~np.any(cover - correct_cover[0], axis=1)).any()
    assert (~np.any(cover - correct_cover[1], axis=1)).any()
    assert (~np.any(cover - correct_cover[2], axis=1)).any()


def test_find_cm_on_clean_am_cm_diff():
    cover = find_cm(ex.am_cm_diff_UDG())
    assert cover.shape == (5, 8)

    correct_cover = ex.am_cm_diff_MCM()

    assert (~np.any(cover - correct_cover[0], axis=1)).any()
    assert (~np.any(cover - correct_cover[1], axis=1)).any()
    assert (~np.any(cover - correct_cover[2], axis=1)).any()
    assert (~np.any(cover - correct_cover[3], axis=1)).any()
    assert (~np.any(cover - correct_cover[4], axis=1)).any()  # not from rule 2


# def test_reduce_rule_1_on_triangle():
#     graph_triangle = np.asarray([
#         [1, 1, 1, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0],
#         [1, 1, 1, 0, 1, 1],
#         [0, 1, 0, 1, 1, 0],
#         [0, 1, 1, 1, 1, 1],
#         [0, 0, 1, 0, 1, 1]])
