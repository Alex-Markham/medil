import numpy as np
import numpy.typing as npt


def nsfd(true_biadj: npt.NDArray, predicted_biadj: npt.NDArray) -> float:
    """Measure distance between predicted and true and structures.

    Parameters
    ----------
    predicted_biadj: learned bipartite directed graph
    true_biadj: true bipartite directed graph

    Returns
    -------
    nsfd: normalized structural Frobenius distance
    """
    true_biadj = true_biadj.astype(int)
    true_wtd_ug = true_biadj.T @ true_biadj
    true_zeros = np.where(true_wtd_ug == 0)
    true_wtd_ug[true_zeros] = -1

    predicted_biadj = predicted_biadj.astype(int)
    predicted_wtd_ug = predicted_biadj.T @ predicted_biadj
    predicted_zeros = np.where(predicted_wtd_ug == 0)
    predicted_wtd_ug[predicted_zeros] = -1

    similarity = np.sum(true_wtd_ug * predicted_wtd_ug)

    cosin_normalizer = np.sqrt((true_wtd_ug**2).sum()) * np.sqrt(
        (predicted_wtd_ug**2).sum()
    )

    nsfd = np.arccos(similarity / cosin_normalizer) / np.pi

    return nsfd


def nshd(
    true_biadj: npt.NDArray,
    *,
    predicted_biadj: npt.NDArray = np.array([]),
    predicted_adj: npt.NDArray = np.array([]),
) -> float:
    """Measure distance between predicted and true and structures.

    Parameters
    ----------
    true_biadj: true bipartite directed graph
    predicted_biadj: learned bipartite directed graph
    predicted_adj: learned mixed graph

    Returns
    -------
    nshd: normalized structural Hamming distance
    """
    if bool(len(predicted_biadj)) == bool(len(predicted_adj)):
        raise ValueError(
            "Must provide `predicted_biadj` or `predicted_adj` but not both."
        )
    elif bool(len(predicted_biadj)):
        predicted_adj = recover_ug(predicted_biadj)

    ug = recover_ug(true_biadj)

    shd = np.logical_xor(ug, predicted_adj).sum()
    n = len(ug)
    nshd = shd / (n**2 - n)
    return nshd


def recover_ug(biadj_mat: npt.NDArray) -> npt.NDArray:
    """Recover the undirected graph from the directed bipartite graph
    Parameters
    ----------
    biadj_mat: learned directed graph

    Returns
    -------
    ug: the recovered undirected graph
    """
    ug = biadj_mat.T @ biadj_mat
    np.fill_diagonal(ug, False)
    return ug
