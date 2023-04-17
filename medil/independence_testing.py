"""Independence testing on samples of random variables."""
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from multiprocessing import Pool

from numpy import linalg as LA
from scipy.stats import chi2

try:
    from dcor import pairwise, distance_correlation as dist_corr

    default_measure = "dcor"
except ImportError:
    default_measure = "pearson"


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


def estimate_UDG(samples, method="dcov_fast", significance_level=0.05):
    num_samps = len(samples)

    if method == "dcov_fast":
        cov, d_bars = dcov(samples)
        crit_val = chi2(1).ppf(1 - significance_level)
        test_val = num_samps * cov / np.outer(d_bars, d_bars)
        udg = test_val >= crit_val
        np.fill_diagonal(udg, False)
    elif method == "g-test":
        pass
    return udg


def dep_con_kernel_one_samp(X, alpha=None):
    num_samps, num_feats = X.shape
    thresh = np.eye(num_feats)
    if alpha is not None:
        thresh[thresh == 0] = (
            chi2(1).ppf(1 - alpha) / num_samps
        )  # critical value corresponding to alpha
        thresh[thresh == 1] = 0
    Z = np.zeros((num_feats, num_samps, num_samps))
    for j in range(num_feats):
        D = squareform(pdist(X[:, j].reshape(-1, 1), "cityblock"))
        # doubly center and standardized:
        Z[j] = ((D - D.mean(0) - D.mean(1).reshape(-1, 1)) / D.mean()) + 1
    F = Z.reshape(num_feats * num_samps, num_samps)
    left = np.tensordot(Z, thresh, axes=([0], [0]))
    left_right = np.tensordot(left, Z, axes=([2, 1], [0, 1]))
    gamma = (F.T @ F) ** 2 - 2 * (left_right) + LA.norm(thresh)  # helper kernel

    diag = np.diag(gamma)
    kappa = gamma / np.sqrt(np.outer(diag, diag))  # cosine similarity
    kappa[kappa > 1] = 1  # correct numerical errors
    return kappa


# note: for this and one_samp, add outputs arg, with options to compute/return Gram matrix, similarity, or distance
def dep_con_kernel_two_samp(samps_1, samps_2, alpha):
    num_samps_1 = len(samps_1)
    samps = np.vstack((samps_1, samps_2))
    full_kappa = dep_con_kernel_one_samp(samps, alpha=None)
    kappa = full_kappa[:num_samps_1]
    kappa = kappa[:, num_samps_1:]
    return kappa
