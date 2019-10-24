from medil.ecc_algorithms import find_clique_min_cover as find_cm
import numpy as np


test_graph_triangle = np.asarray([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1]])

cover = find_cm(graph)
