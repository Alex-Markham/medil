"""Independence testing on samples of random variables."""
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from multiprocessing import Pool, cpu_count

from numpy import linalg as LA
from scipy.stats import chi2

try:
    from dcor.independence import distance_correlation_t_test
    from dcor import pairwise, distance_correlation as dist_corr

    default_measure = "dcor"
except ImportError:
    default_measure = "pearson"
from xicorrelation import xicorr


def hypothesis_test(samples, num_resamples, measure=default_measure, alpha=0.05):
    r"""Performs random permutation tests to estimate independence and
    returns the estimated Undirected Dependency Graph in the form of an
    adjacency matrix.

    Parameters
    ----------
    samples : 2d numpy array of floats or ints
              A :math:`M \times N` matrix with :math:`N` samples of
              :math:`M` random variables.

    num_resamples : int, optional
                    Number of permutations used to calculate p-values.

    measure : str, optional
              Measure of association to use as test statistic. Either
              `pearson` (default if `dcor` package is not installed)
              for a linear measure or `dcor` (default if installed) for
              a nonlinear measure.

    alpha : float, optional
            The threshold on the p-values above which a result is
            considered statistically significant.

    Returns
    -------
    p_values : 2d numpy array of floats
               A square matrix :math:`P`, where :math:`P_{i,j}` is the
               probability of obtaining a result at least as extreme
               as the one given by :math:`N_{i, j}`.

    sample_corr : 2d numpy array of floats
                  A square matrix :math:`N`, where :math:`N_{i,j}` is
                  the measured association value (e.g., correlation)
                  between random variables :math:`R_i` and :math:`R_j`.

    deps : 2d numpy array of bools
           A square matrix `D`, where :math:`D_{i,j}` is true if and
           only if the corresponding random variables :math:`R_i` and
           :math:`R_j` are not estimated to be independent.

    See Also
    --------
    pearson_correlation : Used when ``measure == 'pearson'``.
    distance_correlation : Used when ``measure == 'dcor'``.

    Notes
    -----
    The runtime when using ``distance_correlation`` is nearly cut in
    halve by first generating two permuted samples matrices, ``a`` and
    ``b`` and then calling ``distance_correlation(a, b)`` and using the
    upper and lower triangles, as opposed to calling
    ``distance_correlation(a)`` and ``distance_correlation(b)``
    separately while using only one triangle from each. A similar trick
    could be employed in ``pearson_correlation``, but it reduces the
    runtime by less than 10%. Another (mathematically) similar trick
    would be to compute the correlation on ``np.vstack((a, b))``, but
    this also is less than a 10% improvement.

    """
    if measure == "pearson":
        compute_corr = pearson_correlation
    elif measure == "dcor":
        compute_corr = distance_correlation
    else:
        raise ValueError("{} is not a supported measure of association".format(measure))

    sample_corr = compute_corr(samples)

    # initialize aux vars used in loop
    p_values = np.zeros(sample_corr.shape)
    num_loops = num_resamples if measure != "dcor" else int(np.ceil(num_resamples / 2))
    for _ in range(num_loops):
        perm_corr = compute_corr(samples, perm=True)
        p_values += np.array(perm_corr >= sample_corr, int)

    p_values += p_values.T
    p_values /= num_loops if measure != "dcor" else 2 * num_loops

    deps = p_values <= alpha  # reject null hypothesis of independence

    return p_values, sample_corr, deps


def distance_correlation(samples, perm=False):
    r"""Compute distance correlation on samples set (or on permuted samples
    set, if ``perm`` is ``True).

    Parameters
    ----------
    samples : 2d numpy array of floats or ints
              A :math:`M \times N` matrix with :math:`N` samples of
              :math:`M` random variables.

    perm : bool, optional
           Whether distance correlation is computed on permuted or
           original samples.

    Returns
    -------
    2d numpy array of floats
        A square matrix :math:`C`, where :math:`C_{i,j}` is
        (if ``perm``, a sample from the null distribution of)
        the distance correlation between random variables :math:`R_i`
        and :math:`R_j`.

    """
    try:
        pairwise
    except NameError:
        raise ImportError(
            "Install the dcor package to use this feature: `pip install dcor`."
        )
    if not perm:
        with Pool() as pool:
            corr = pairwise(dist_corr, samples, pool=pool)
    else:
        samples_permed = permute_within_rows(samples)
        samples_permed_2 = permute_within_rows(samples)
        with Pool() as pool:
            corr = pairwise(dist_corr, samples_permed, samples_permed_2, pool=pool)
    return corr


def pearson_correlation(samples, perm=False):
    r"""Computes Pearson product-moment correlation coefficient on samples
    set (or on permuted samples set, if ``perm`` is ``True).

    Parameters
    ----------
    samples : 2d numpy array of floats or ints
              A :math:`M \times N` matrix with :math:`N` samples of
              :math:`M` random variables.

    perm : bool, optional
           Whether Pearson correlation is computed on permuted or
           original samples.

    Returns
    -------
    2d numpy array of floats
        A square matrix :math:`C`, where :math:`C_{i,j}` is
        (if ``perm``, a sample from the null distribution of)
        the Pearson correlation between random variables :math:`R_i`
        and :math:`R_j`.

    """
    if not perm:
        corr = np.corrcoef(samples)
    else:
        samples_permed = permute_within_rows(samples)
        corr = np.corrcoef(samples, samples_permed)
    return corr


def permute_within_rows(x):
    """Randomly rearrange values according to column index without
    changing row index.

    Parameters
    ----------
    x : 2d numpy array

    Returns
    -------
    2d numpy array
        Reordered copy of input.

    """
    # get random new index for col of each element
    col_idx = np.random.sample(x.shape).argsort(axis=1)

    # keep the row index the same
    row_idx = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T

    # apply the permutaton matrix to permute x
    return x[row_idx, col_idx]


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
            udg = (p_vals < significance_level)
    np.fill_diagonal(udg, False)
    return udg, p_vals


def dcor_test(x_y):
    x, y = x_y
    return distance_correlation_t_test(x, y).pvalue


def xicor_test(x_y):
    x, y = x_y
    xi, pvalue = xicorr(x, y)
    return pvalue
