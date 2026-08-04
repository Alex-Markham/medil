"""Evaluation metrics for learned MeDIL causal model structures."""

import numpy as np
import numpy.typing as npt


def sfd(
    true_biadj: npt.NDArray,
    predicted_biadj: npt.NDArray,
    to_return: str = "raw",
) -> int | float | tuple[int, float]:
    """Shared factor distance sums difference of latent parents.

    For a binary biadjacency matrix B, consider U = B'B, where U_ij
    counts the number of parents nodes i and j have in common (so U_ii
    is the assignment number, and diag(U) is a sufficient statistic
    for the graph under the 1-pure-child assumption). sfd(B_1, B_2) is
    the sum of the differences between U_ij for B_1 and B_2 (without
    double-counting for U_ji).

    Parameters
    ----------
    true_biadj : ndarray
        Ground-truth biadjacency matrix.
    predicted_biadj : ndarray
        Learned biadjacency matrix.
    to_return : str, optional
        ``"raw"`` (default), ``"normalized"``, or ``"both"``.

    Returns
    -------
    int or float or tuple
        Raw sfd (int) if ``to_return='raw'``, normalized nsfd (float) if
        ``to_return='normalized'``, or ``(sfd, nsfd)`` if ``to_return='both'``.
    """
    true_biadj = true_biadj.astype(int)
    true_wtd_ug = true_biadj.T @ true_biadj

    predicted_biadj = predicted_biadj.astype(int)
    predicted_wtd_ug = predicted_biadj.T @ predicted_biadj

    sfd = np.abs(np.triu(true_wtd_ug - predicted_wtd_ug)).sum()

    if to_return == "raw":
        return sfd

    true_zeros = np.where(true_wtd_ug == 0)
    true_wtd_ug[true_zeros] = -1

    predicted_zeros = np.where(predicted_wtd_ug == 0)
    predicted_wtd_ug[predicted_zeros] = -1

    similarity = np.sum(true_wtd_ug * predicted_wtd_ug)

    cosin_normalizer = np.sqrt((true_wtd_ug**2).sum()) * np.sqrt(
        (predicted_wtd_ug**2).sum()
    )

    nsfd = np.arccos(similarity / cosin_normalizer) / np.pi

    match to_return:
        case "normalized":
            return nsfd
        case "both":
            return sfd, nsfd
        case _:
            raise ValueError("`to_return` should be 'raw', 'normalized', or 'both'")


def _shd(
    true_biadj: npt.NDArray,
    *,
    predicted_biadj: npt.NDArray = np.array([]),
    predicted_adj: npt.NDArray = np.array([]),
    to_return: str = "raw",
) -> int | float | tuple[int, float]:
    """Structural Hamming distance counts number of incorrect arrowheads/tails.

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
        predicted_adj = _recover_ug(predicted_biadj)

    ug = _recover_ug(true_biadj)

    shd = np.logical_xor(ug, predicted_adj).sum()

    if to_return == "raw":
        return shd

    n = len(ug)
    nshd = shd / (n**2 - n)

    match to_return:
        case "normalized":
            return nshd
        case "both":
            return shd, nshd
        case _:
            raise ValueError("`to_return` should be 'raw', 'normalized', or 'both'")


def _recover_ug(biadj_mat: npt.NDArray) -> npt.NDArray:
    ug = biadj_mat.T @ biadj_mat
    np.fill_diagonal(ug, False)
    return ug
