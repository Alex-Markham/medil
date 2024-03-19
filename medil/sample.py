"""Generate random minimum MeDIL causal model graph or parameters."""
import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

from .models import MedilCausalModel


def mcm(
    parameterization: str = "gauss", biadj: None | npt.NDArray = None, **kwargs
) -> MedilCausalModel:
    if biadj is None:
        biadj = _biadj(**kwargs)
    return MedilCausalModel()


def biadj(
    num_meas: int,
    density: float = 0.2,
    one_pure_child: bool = True,
    num_latent: int = 0,
) -> npt.NDArray:
    """Randomly generate biadjacency matrix for graphical minMCM."""
    if one_pure_child:
        return rand_1pc(num_meas)
    else:
        if num_latent != 0:
            msg = "`num_latent` can only be specified when `one_pure_child==True`."
            raise ValueError(msg)
        return rand_er(density)


_biadj = biadj


def rand_1pc(num_obs, effect_prob=None, rng=default_rng(0)):
    num_latent = rng.integers(1, num_obs)
    biadj_mat = np.zeros((num_latent, num_obs), bool)
    biadj_mat[:, :num_latent] = np.eye(num_latent)
    if effect_prob is None:
        effect_prob = rng.random()
    max_num_edges = (num_obs - num_latent) * num_latent
    num_edges = np.round(max_num_edges * effect_prob).astype(int)
    edges = np.zeros(max_num_edges, bool)
    edges[:num_edges] = True
    edges = rng.permutation(edges).reshape(num_latent, num_obs - num_latent)
    # effect_prob is a conditional sparsity of the biadj_mat, given
    # the number of obs and latent and that it satisfies 1pc
    biadj_mat[:, num_latent:] = edges
    return rng.permutation(biadj_mat, axis=1)


def rand_er(edge_prob):
    """Generate minMCM from Erdős–Rényi random undirected graph
    over observed variables."""
    # ER random graph
    udg = np.zeros((num_obs, num_obs), bool)
    max_edges = (num_obs * (num_obs - 1)) // 2
    num_edges = np.round(edge_prob * max_edges).astype(int)
    edges = np.ones(max_edges)
    edges[num_edges:] = 0
    udg[np.triu_indices(num_obs, k=1)] = rng.permutation(edges)
    udg += udg.T
    np.fill_diagonal(udg, True)
    udg = udg

    # find latent connections (minimum edge clique cover)
    biadj_mat = find_cm(udg)
    biadj_mat = biadj_mat.astype(bool)
    num_latent = len(biadj_mat)
    return biadj_mat
