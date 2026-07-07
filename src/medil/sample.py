"""Generate random minimum MeDIL causal model graph or parameters."""

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

from .ecc_algorithms import _find_clique_min_cover
from .models import GaussianMCM, NeuroCausalFactorAnalysis


def mcm(
    rng: np.random.Generator = default_rng(0),
    parameterization: str = "Gaussian",
    biadj: npt.NDArray = np.array([]),
    **kwargs,
) -> GaussianMCM | NeuroCausalFactorAnalysis:
    """Randomly generate a minimum MeDIL causal model with parameters.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Random number generator. Default is ``default_rng(0)``.
    parameterization : str, optional
        Either ``"Gaussian"`` (default) for a linear Gaussian model or
        ``"VAE"`` for a randomly initialized masked VAE model.
    biadj : ndarray, optional
        Biadjacency matrix to use. If empty (default), one is generated
        randomly using :func:`biadj` with any extra keyword arguments.
    **kwargs
        Additional keyword arguments passed to :func:`biadj` when
        generating a random biadjacency matrix.

    Returns
    -------
    GaussianMCM or NeuroCausalFactorAnalysis
        A fitted model with randomly generated structure and parameters.
    """
    if biadj.size == 0:
        biadj = _biadj(rng=rng, **kwargs)
    if parameterization == "Gaussian":
        mcm = GaussianMCM(biadj=biadj, rng=rng)
        params = mcm.parameters

        num_edges = biadj.sum()
        weights = (rng.random(num_edges) * 1.5) + 0.5
        weights[rng.choice((True, False), num_edges)] *= -1
        params.biadj_weights = np.zeros_like(biadj, float)
        params.biadj_weights[biadj] = weights

        num_meas = biadj.shape[1]
        params.error_means = rng.random(num_meas) * 2
        params.error_means[rng.choice((True, False), num_meas)] *= -1

        params.error_variances = (rng.random(num_meas) * 1.5) + 0.5

    elif parameterization == "VAE":
        try:
            import torch
            from ._vae import VariationalAutoencoder
        except ImportError:
            raise ImportError(
                "parameterization='VAE' requires PyTorch. "
                "Install it with: pip install medil[ncfa]"
            )
        model = NeuroCausalFactorAnalysis(biadj=biadj, rng=rng)
        num_latent, num_meas = biadj.shape
        biadj_tensor = torch.tensor(biadj.T, dtype=torch.float32)
        vae = VariationalAutoencoder(
            num_latent=num_latent,
            num_meas=num_meas,
            num_hidden_layers=model.hyperparams["num_hidden_layers"],
            latent_width=model.hyperparams["latent_width"],
            meas_width=model.hyperparams["meas_width"],
            biadj=biadj_tensor,
            encoder_hidden_dim=model.hyperparams["encoder_hidden_dim"],
        ).to(model.device)
        model.parameters.vae = vae
        return model

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
        biadj = _find_clique_min_cover(udg).astype(bool)

    return biadj


_biadj = biadj
