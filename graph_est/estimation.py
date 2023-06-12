from medil.ecc_algorithms import find_clique_min_cover
from medil.ecc_algorithms import find_heuristic_clique_cover
from medil.independence_testing import estimate_UDG
import numpy as np


def estimation(samples_out, heuristic=False, method="dcov_fast", alpha=0.05):
    """ Perform estimations of the shd and number of reconstructed latent
    Parameters
    ----------
    samples_out: output samples
    heuristic: whether to use the heuristic solver
    method: method for udg estimation
    alpha: significance level

    Returns
    -------
    ud_graph: learned undirected graph
    biadj_mat_recon: learned directed graph in the form of adjacency matrix
    """

    # step 1: estimate UDG
    ud_graph, p_vals = estimate_UDG(samples_out, method=method, significance_level=alpha)
    np.fill_diagonal(ud_graph, val=True)

    # step 2: learn graphical MCM
    if heuristic:
        biadj_mat_recon = find_heuristic_clique_cover(ud_graph)
    else:
        biadj_mat_recon = find_clique_min_cover(ud_graph)

    return biadj_mat_recon
