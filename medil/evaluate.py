import itertools

import numpy as np
import numpy.typing as npt


def sfd(predicted_biadj, true_biadj):
    """Perform analysis of the distances between true and reconstructed structures
    Parameters
    ----------
    biadj_mat: input directed graph
    biadj_mat_recon: learned directed graph in the form of adjacency matrix

    Returns
    -------
    sfd: squared Frobenius distance (bipartite graph)
    ushd: structural hamming distance (undirected graph)
    """

    # ushd = shd_func(recover_ug(biadj_mat), recover_ug(biadj_mat_recon))
    ug = recover_ug(true_biadj)
    ug_recon = recover_ug(predicted_biadj)

    ushd = np.triu(np.logical_xor(ug, ug_recon), 1).sum()

    true_biadj = true_biadj.astype(int)
    predicted_biadj = predicted_biadj.astype(int)

    wtd_ug = true_biadj.T @ true_biadj
    wtd_ug_recon = predicted_biadj.T @ predicted_biadj

    sfd = ((wtd_ug - wtd_ug_recon) ** 2).sum()

    return sfd, ushd


def recover_ug(biadj_mat):
    """Recover the undirected graph from the directed graph
    Parameters
    ----------
    biadj_mat: learned directed graph

    Returns
    -------
    ug: the recovered undirected graph
    """

    # get the undirected graph from the directed graph
    ug = biadj_mat.T @ biadj_mat
    np.fill_diagonal(ug, False)

    return ug


def min_perm_squared_l2_dist(predicted_W: npt.NDArray, true_W: npt.NDArray):
    def perm_squared_l2_dist(perm):
        return np.sum((predicted_W[perm] - true_W) ** 2)

    def pair(perm):
        return perm, perm_squared_l2_dist(perm)

    perms = itertools.permutations(range(len(predicted_W)))

    pairs = map(pair, perms)

    opt_perm, min_dist = min(pairs, key=lambda pair: pair[1])

    return opt_perm, min_dist
