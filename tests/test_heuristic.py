import numpy as np
from medil.ecc_algorithms import find_heuristic_clique_cover
import medil.examples as ex


def test_find_heuristic_clique_cover_triangle():
    the_cover = find_heuristic_clique_cover(ex.triangle.UDG)

    c_1, c_2, c_3 = np.array(
        [
            [True, True, True, False, False, False],
            [False, True, False, True, True, False],
            [False, False, True, False, True, True],
        ]
    )

    for c in c_1, c_2, c_3:
        assert (c == the_cover).all(1).any()
    assert len(the_cover) == 3


def test_find_heuristic_clique_cover_complete():
    the_cover = find_heuristic_clique_cover(np.ones((5, 5), int))
    assert (the_cover == np.ones((1, 5), bool)).all()
