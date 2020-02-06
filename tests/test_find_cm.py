import numpy as np
from medil.ecc_algorithms import find_clique_min_cover as find_cm
from medil.graph import UndirectedDependenceGraph
from medil.ecc_algorithms import branch


# Here are some integration tests.
def test_find_cm_on_3_cycle():
    cycle_3 = np.ones((3, 3), dtype=int)
    cover = find_cm(cycle_3, True)
    assert cover.shape==(1, 3)
    assert [1, 1, 1] in cover


# def test_reduction_rule_1_on_3cycle_plus_isolated():
#     graph = np.zeros((4, 4), dtype=int)  # init
#     graph[1:4, 1:4] = 1         # add 3cycle
#     graph[0, 0] = 1             # add isolated vert

#     cover = find_cm(graph, verbose=True)
    
#     assert cover.shape==(1, 4)
#     assert [0, 1, 1, 1] in cover


# def test_find_cm_on_triangle():
#     graph_triangle = np.asarray([
#         [1, 1, 1, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0],
#         [1, 1, 1, 0, 1, 1],
#         [0, 1, 0, 1, 1, 0],
#         [0, 1, 1, 1, 1, 1],
#         [0, 0, 1, 0, 1, 1]])

#     cover = find_cm(graph_triangle)
#     assert cover.shape==(3, 6)
#     assert [0, 0, 1, 0, 1, 1] in cover
#     assert [0, 1, 0, 1, 1, 0] in cover
#     assert [1, 1, 1, 0, 0, 0] in cover


# def test_find_cm_on_clean_am_cm_diff():
#     graph  = np.asarray([
#         [1, 1, 1, 0, 1, 0, 1, 0],
#         [1, 1, 1, 0, 1, 1, 1, 1],
#         [1, 1, 1, 0, 1, 1, 1, 1],
#         [0, 0, 0, 1, 0, 1, 1, 1],
#         [1, 1, 1, 0, 1, 1, 0, 1],
#         [0, 1, 1, 1, 1, 1, 0, 1],
#         [1, 1, 1, 1, 0, 0, 1, 1],
#         [0, 1, 1, 1, 1, 1, 1, 1]])

#     cover = find_cm(graph, verbose=True)
#     assert cover.shape==(5, 8)
#     assert [1, 1, 1, 0, 1, 0, 0, 0] in cover
#     assert [1, 0, 0, 1, 0, 0, 1, 0] in cover
#     assert [0, 1, 1, 0, 1, 1, 0, 1] in cover
#     assert [0, 0, 0, 1, 0, 1, 0, 1] in cover
#     assert [0, 1, 1, 0, 0, 0, 1, 1] in cover


# def test_real_data():
#     results = np.load("/home/alex/Projects/mcm_paper/uai_2020/data_analysis/monte_carlo_test_results_1000.npz")
#     all_deps = results['deps']

#     deps = all_deps[2:63, 2:63]

#     cover = find_cm(deps.astype(int), verbose=True)

# test_real_data()

# Here are unit tests.

def test_make_aux_on_triangle():
    graph_triangle = np.asarray([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1]])
    graph = UndirectedDependenceGraph(graph_triangle)
    graph.make_aux()

    assert (graph.common_neighbors[[2, 3, 4, 8, 9, 13]]==0).all()

def test_reduce_rule_1():
    pass
    # random graphs then add isolated verts and remove with rule_1
    # init new UDG with non-isolated verts and make_auk then compare values with assert

    
