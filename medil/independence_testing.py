"""Independence testing on samples of random variables."""
import numpy as np

from multiprocessing import Pool

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
