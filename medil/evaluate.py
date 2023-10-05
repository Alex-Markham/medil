import numpy as np


def sfd(biadj_mat, biadj_mat_recon):
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
    ug = recover_ug(biadj_mat)
    ug_recon = recover_ug(biadj_mat_recon)

    ushd = np.triu(np.logical_xor(ug, ug_recon), 1).sum()

    biadj_mat = biadj_mat.astype(int)
    biadj_mat_recon = biadj_mat_recon.astype(int)

    wtd_ug = biadj_mat.T @ biadj_mat
    wtd_ug_recon = biadj_mat_recon.T @ biadj_mat_recon

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
