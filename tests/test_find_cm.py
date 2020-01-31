import numpy as np
from medil.ecc_algorithms import find_clique_min_cover as find_cm


def test_find_cm_on_3_cycle():
    cycle_3 = np.ones((3, 3), dtype=int)
    cover = find_cm(cycle_3, verbose=True)
    assert cover.shape==(1, 3)
    assert (cover==[[1, 1, 1]]).all()


def test_find_cm_on_triangle():
    graph_triangle = np.asarray([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1]])

    cover = find_cm(graph_triangle, verbose=True)
    assert cover.shape==(3, 6)
    assert [0, 0, 1, 0, 1, 1] in cover
    assert [0, 1, 0, 1, 1, 0] in cover
    assert [1, 1, 1, 0, 0, 0] in cover


def test_find_cm_on_clean_am_cm_diff():
    graph  = np.asarray([
        [1, 1, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 0, 1],
        [0, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1]])

    cover = find_cm(graph, verbose=True)
    assert cover.shape==(5, 8)
    assert [1, 1, 1, 0, 1, 0, 0, 0] in cover
    assert [1, 0, 0, 1, 0, 0, 1, 0] in cover
    assert [0, 1, 1, 0, 1, 1, 0, 1] in cover
    assert [0, 0, 0, 1, 0, 1, 0, 1] in cover
    assert [0, 1, 1, 0, 0, 0, 1, 1] in cover


# def test_real_data():
#     results = np.load("/home/alex/Data/exploratory/monte_carlo_test_results_1000.npz")
#     all_deps = results['deps']

#     deps = all_deps[2:63, 2:63]

#     cover = find_cm(deps.astype(int), verbose=True)

# import time

# start = time.time()
# for _ in range(1000):
#     cm = find_cm(test_graph_triangle)
# end = time.time()
# print((end - start) / 1000)
