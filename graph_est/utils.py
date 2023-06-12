import numpy as np
import cdt


def permute_graph(biadj_mat_recon, perm):
    """Find permutation of the graph
    Parameters
    ----------
    biadj_mat_recon: learned directed adjacency matrix
    perm: number of observed variables

    Returns
    -------
    biadj_mat_recon: learned directed adjacency matrix with a given permutation
    """

    biadj_mat_perm = biadj_mat_recon[perm, :]

    return biadj_mat_perm


def expand_recon(biadj_mat_recon, num_obs, num_latent):
    """Expand the reconstructed directed graph
    Parameters
    ----------
    biadj_mat_recon: learned directed adjacency matrix
    num_obs: number of observed variables
    num_latent: number of latent variables

    Returns
    -------
    biadj_mat_recon_: expanded learned directed graph
    """

    num_latent_recon = biadj_mat_recon.shape[0]
    biadj_mat_recon_ = np.zeros((num_latent, num_obs))
    biadj_mat_recon_[:num_latent_recon, :] = biadj_mat_recon

    return biadj_mat_recon_


def contract_recon(biadj_mat_recon, comb):
    """Contract the reconstructed directed graph
    Parameters
    ----------
    biadj_mat_recon: learned directed adjacency matrix
    comb: combinations of the latent variables to be selected

    Returns
    -------
    biadj_mat_recon_: contracted learned directed graph
    """

    biadj_mat_recon_ = biadj_mat_recon[comb, :]

    return biadj_mat_recon_


def shd_func(g1, g2):
    """Estimating structural hamming distance given two graphs with the same shape
    Parameters
    ----------
    g1: graph 1 for evaluating the SHD
    g2: graph 2 for evaluating the SHD

    Returns
    -------
    shd: structural hamming distance
    """

    if not g1.shape == g2.shape:
        raise ValueError("Mismatch between the shape of graph 1 and graph 2")

    shd = cdt.metrics.SHD(g1, g2)

    return shd


def generate_linspace(lin_min, lin_max, size):
    """Generate linspace given the minimal and maximal value inputs
    Parameters
    ----------
    lin_min: minimum value
    lin_max: maximum value
    size: size of the dataset
    Returns
    -------

    """

    linspace = np.exp(np.linspace(np.log(lin_min), np.log(lin_max), size))
    linspace = np.array(sorted(set(np.round(linspace)))).astype(int)

    return linspace
