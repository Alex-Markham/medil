"""Independence testing on samples of random variables."""

from typing import NamedTuple, Optional

from dcor.independence import distance_correlation_t_test
from multiprocessing import Pool, cpu_count
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2, norm, rankdata


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
        pass
    else:
        p_vals = np.zeros((num_feats, num_feats), float)
        idxs, jdxs = np.triu_indices(num_feats, 1)
        zipped = zip(idxs, jdxs)
        sample_iter = (sample[:, i_j].T for i_j in zipped)
        if method == "dcov_big":
            test = dcor_test
        elif method == "xicor":
            test = xicor_test
        with Pool(max(1, int(0.75 * cpu_count()))) as p:
            p_vals[idxs, jdxs] = p_vals[jdxs, idxs] = np.fromiter(
                p.imap(test, sample_iter, 100), float
            )
            udg = p_vals < significance_level
    np.fill_diagonal(udg, False)
    return udg, p_vals


def dcor_test(x_y):
    x, y = x_y
    return distance_correlation_t_test(x, y).pvalue


def xicor_test(x_y):
    x, y = x_y
    xi, pvalue = xicorr(x, y)
    return pvalue


# The following is a modification of the xicorrelation source code,
# Copyright 2021 Nikolay Novik (https://github.com/jettify),
# for compatibility with numpy 2.0+
class _XiCorr(NamedTuple):
    xi: float
    fr: npt.NDArray[np.float64]
    cu: float


class XiCorrResult(NamedTuple):
    correlation: float
    pvalue: Optional[float]


def _xicorr(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> _XiCorr:
    # Ported from original R implementation.
    # https://github.com/cran/XICOR/blob/master/R/calculateXI.R
    n = x.size
    PI = rankdata(x, method="average")
    fr = rankdata(y, method="average") / n
    gr = rankdata(-y, method="average") / n
    cu = np.mean(gr * (1 - gr))
    A1 = np.abs(np.diff(fr[np.argsort(PI, kind="quicksort")])).sum() / (2 * n)
    xi = 1.0 - A1 / cu
    return _XiCorr(xi, fr, cu)


def xicorr(x: npt.ArrayLike, y: npt.ArrayLike, ties: bool = True) -> XiCorrResult:
    """Compute the cross rank increment correlation coefficient xi [1].

    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they
        will be flattened to 1-D.
    ties : bool, optional
        If ties is True, the algorithm assumes that the data has ties and
        employs the more elaborated theory for calculating s.d. and P-value.
        Otherwise, it uses the simpler theory. There is no harm in putting
        ties = True even if there are no ties.

    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       P-values computed by the asymptotic theory.

    See Also
    --------
    spearmanr : Calculates a Spearman rank-order correlation coefficient.


    Example:
        >>> from xicorrelation import xicorr
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [1, 4, 9, 16, 25]
        >>> xi, pvalue = xicorr(x, y)
        >>> print(xi, pvalue)

    References
    ----------
    .. [1] Chatterjee, S., "A new coefficient of correlation",
           https://arxiv.org/abs/1909.10140, 2020.

    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> x2 = [1, 4, 7, 1, 0]
    >>> xi, p_value, _ = xicorr(x1, x2)
    >>> tau
    -0.47140452079103173
    >>> p_value
    0.2827454599327748
    """
    # https://git.io/JSIlN
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError(
            "All inputs to `xicorr` must be of the same "
            f"size, found x-size {x.size} and y-size {y.size}"
        )
    elif not x.size or not y.size:
        # Return NaN if arrays are empty
        return XiCorrResult(np.nan, np.nan)

    r = _xicorr(x, y)
    xi = r.xi
    fr = r.fr
    CU = r.cu

    pvalue: Optional[float] = None
    # https://git.io/JSIlM
    n = x.size
    if not ties:
        # sd = np.sqrt(2.0 / (5.0 * n))
        pvalue = 1.0 - norm.cdf(np.sqrt(n) * xi / np.sqrt(2.0 / 5.0))
    else:
        qfr = np.sort(fr)
        ind = np.arange(1, n + 1)
        ind2 = 2 * n - 2 * ind + 1

        ai = np.mean(ind2 * qfr * qfr) / n
        ci = np.mean(ind2 * qfr) / n
        cq = np.cumsum(qfr)
        m = (cq + (n - ind) * qfr) / n
        b = np.mean(m**2)
        v = (ai - 2.0 * b + ci**2) / (CU**2)

        # sd = np.sqrt(v / n)
        pvalue = 1.0 - norm.cdf(np.sqrt(n) * xi / np.sqrt(v))
    return XiCorrResult(xi, pvalue)
