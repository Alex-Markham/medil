"""Independence testing on samples of random variables."""

from multiprocessing import Pool, cpu_count
from typing import NamedTuple, Optional

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chatterjeexi, chi2


def dcov(samples):
    r"""Compute sample distance covariance matrix.

    Parameters
    ----------
    samples : 2d numpy array of floats
              A :math:`N \times M` matrix with :math:`N` samples of
              :math:`M` random variables.
    Returns
    -------
    2d numpy array
        A square matrix :math:`C`, where :math:`C_{i,j}` is the sample
        distance covariance between random variables :math:`R_i` and
        :math:`R_j`.

    Notes
    -----
    Trades time complexity for space complexity, so it can be too
    memory-intensive for larger datasets, in which case recommend to
    use xicor or distance_correlation_t_test from the dcor package.
    """
    num_samps, num_feats = samples.shape
    num_pairs = num_samps * (num_samps - 1) // 2
    dists = np.zeros((num_feats, num_pairs))
    d_bars = np.zeros(num_feats)
    # compute doubly centered distance matrix for every feature:
    for feat_idx in range(num_feats):
        n = num_samps
        t = np.tile
        # raw distance matrix:
        d = squareform(pdist(samples[:, feat_idx].reshape(-1, 1), "cityblock"))
        # doubly centered:
        d_bar = d.mean()
        d -= t(d.mean(0), (n, 1)) + t(d.mean(1), (n, 1)).T - t(d_bar, (n, n))
        d = squareform(d, checks=False)  # ignore assymmetry due to numerical error
        dists[feat_idx] = d
        d_bars[feat_idx] = d_bar
    return dists @ dists.T / num_samps**2, d_bars


def estimate_UDG(sample, method="dcov_fast", significance_level=0.05):
    samp_size, num_feats = sample.shape

    if isinstance(method, np.ndarray):
        p_vals = method
        udg = p_vals < significance_level
    elif method == "dcov_fast":
        cov, d_bars = dcov(sample)
        crit_val = chi2(1).ppf(1 - significance_level)
        test_val = samp_size * cov / np.outer(d_bars, d_bars)
        udg = test_val >= crit_val
        p_vals = None
    elif method == "g-test":
        raise NotImplementedError("g-test is not yet implemented")
    else:
        p_vals = np.zeros((num_feats, num_feats), float)
        idxs, jdxs = np.triu_indices(num_feats, 1)
        zipped = zip(idxs, jdxs)
        sample_iter = (sample[:, i_j].T for i_j in zipped)
        # if method == "dcov_big":
        #     can use distance_correlation_t_test from dcor package
        if method == "xicor":
            test = xicor_test
        with Pool(max(1, int(0.75 * cpu_count()))) as p:
            p_vals[idxs, jdxs] = p_vals[jdxs, idxs] = np.fromiter(
                p.imap(test, sample_iter, 100), float
            )
            udg = p_vals < significance_level
    np.fill_diagonal(udg, False)
    return udg, p_vals


def xicor_test(x_y):
    x, y = x_y
    xi, pvalue = xicorr(x, y)
    return pvalue


class XiCorrResult(NamedTuple):
    correlation: float
    pvalue: Optional[float]


def xicorr(x: npt.ArrayLike, y: npt.ArrayLike, ties: bool = True) -> XiCorrResult:
    """Compute Chatterjee's xi correlation coefficient.

    Parameters
    ----------
    x, y : array_like
        Arrays of the same length. Flattened to 1-D if needed.
    ties : bool, optional
        If True (default), uses the variance formula for tied data.
        There is no harm in leaving this True even when there are no ties.

    Returns
    -------
    correlation : float
        Chatterjee's xi correlation coefficient.
    pvalue : float
        P-value computed by the asymptotic theory.

    References
    ----------
    Chatterjee, S., "A new coefficient of correlation",
    Journal of the American Statistical Association, 116(536), 2021.
    https://doi.org/10.1080/01621459.2020.1758115

    Examples
    --------
    >>> x1 = [12, 2, 1, 12, 2]
    >>> x2 = [1, 4, 7, 1, 0]
    >>> xi, pvalue = xicorr(x1, x2)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size:
        raise ValueError(
            "All inputs to `xicorr` must be of the same "
            f"size, found x-size {x.size} and y-size {y.size}"
        )
    result = chatterjeexi(x, y, y_continuous=not ties)
    return XiCorrResult(result.statistic, result.pvalue)
