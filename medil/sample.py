"""Generate random minimum MeDIL causal model graph or parameters."""
import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

from .models import MedilCausalModel
from .ecc_algorithms import find_clique_min_cover


def mcm(
    rng: np.random.Generator = default_rng(0),
    parameterization: str = "gauss",
    biadj: None | npt.NDArray = None,
    **kwargs,
) -> MedilCausalModel:
    if biadj is None:
        biadj = _biadj(**kwargs)
    if parameterization == "gauss":
        mcm = MedilCausalModel(biadj)

        num_edges = biadj.sum()
        weights = (rng.random(num_edges) * 1.5) + 0.5
        weights[rng.choice((True, False), num_edges)] *= -1
        biadj_weights = np.zeros_like(biadj, float)
        biadj_weights[biadj] = weights
        mcm.biadj_weights = biadj_weights

        num_meas = biadj.shape[1]
        error_means = rng.random(num_meas) * 2
        error_means[rng.choice((True, False), num_meas)] *= -1
        mcm.error_means = error_means

        mcm.error_variances = (rng.random(num_meas) * 1.5) + 0.5

    elif parameterization == "vae":
        raise (NotImplementedError)

    else:
        raise ValueError(f"Parameterization '{parameterization}' is invalid.")

    return mcm


def biadj(
    num_meas: int,
    density: float = 0.2,
    one_pure_child: bool = True,
    num_latent: int = 0,
    rng: np.random.Generator = default_rng(0),
) -> npt.NDArray:
    """Randomly generate biadjacency matrix for graphical minMCM."""
    if one_pure_child:
        """Define a maximum independent set of size `num_latent`, and then grow these into a minimum edge clique cover with average max clique size `2 + (num_meas - num_latent) * density`."""
        if num_latent == 0:
            num_latent = rng.integers(1, num_meas)
        if density is None:
            density = rng.random()

        # specify pure children/independent set
        biadj = np.zeros((num_latent, num_meas), bool)
        biadj[:, :num_latent] = np.eye(num_latent)

        # every child gets a parent; specifically L_0, until the
        # within-column perm below using np.permuted
        biadj[0, num_latent:] = True

        # randomly fill in remaining density * (num_meas - num_latent)
        # * (num_latent - 1) edges
        max_num_edges = (num_meas - num_latent) * (num_latent - 1)
        num_edges = np.round(max_num_edges * density).astype(int)

        edges = np.zeros(max_num_edges, bool)
        edges[:num_edges] = True
        edges = rng.permutation(edges).reshape(num_latent - 1, num_meas - num_latent)

        biadj[1:][:, num_latent:] = edges

        nonpure_children = biadj[:, num_latent:]
        biadj[:, num_latent:] = rng.permuted(nonpure_children, axis=0)

        # change child order, so pure children aren't first
        biadj = rng.permutation(biadj, axis=1)

    else:
        """Generate minMCM from Erdős–Rényi random undirected graph
        over observed variables."""
        if num_latent != 0:
            msg = "`num_latent` can only be specified when `one_pure_child==True`."
            raise ValueError(msg)

        udg = np.zeros((num_meas, num_meas), bool)

        max_edges = (num_meas * (num_meas - 1)) // 2
        num_edges = np.round(density * max_edges).astype(int)

        edges = np.ones(max_edges)
        edges[num_edges:] = 0

        udg[np.triu_indices(num_meas, k=1)] = rng.permutation(edges)
        udg += udg.T
        np.fill_diagonal(udg, True)

        # find latent connections (minimum edge clique cover)
        biadj = find_clique_min_cover(udg).astype(bool)

    return biadj


_biadj = biadj
