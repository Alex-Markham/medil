import numpy as np
from medil.ecc_algorithms import find_clique_min_cover as find_cm


test_graph_triangle = np.asarray([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1]])

cover = find_cm(test_graph_triangle)
print(cover)

# import time

# start = time.time()
# for _ in range(1000):
#     cm = find_cm(test_graph_triangle)
# end = time.time()
# print((end - start) / 1000)
