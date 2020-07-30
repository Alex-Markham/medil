import numpy as np

from multiprocessing import Pool
try:
    from dcor import pairwise, distance_correlation as distcorr
except ImportError:
    print("""With your current packages, only Pearson correlation is available for independence testing. For a nonlinear measure, install the dcor package.""")
    

def dependencies(null_hyp, iota=0.1, p_values, alpha=0.05):
    """Returns the estimated Undirected Dependency Graph in the form of an adjacency matrix.

    Parameters
    ----------
    null_hyp : 2d numpy array of floats
               A square matrix `N`, where :math:`N_{i,j}` is the measured association value (e.g., correlation) between random variables :math:`R_i` and :math:`R_j`.

    iota : float, optional
           The threshold on the measure of association below which two random variables are considered indendent.

    p_values : 2d numpy array of floats
               A square matrix `P`, where :math:`P_{i,j}` is the probability of obtaining a result at least as extreme as the one given by :math:`N_{i, j}`.
    
    alpha : float, optional
            The threshold on the p-values above which a result is considered statistically significant.

    Returns
    -------
    2d numpy array of bools
        A square matrix `D`, where :math:`D_{i,j}` is true if and only if the corresponding random variables :math:`R_i` and :math:`R_j` are estimated to be dependent.

    """
    null_indep = null_hyp <= iota
    accept_null = p_values >= alpha
    independencies = null_indep & accept_null
    return ~independencies     # dependencies


def hypothesis_test(data, num_resamples, measure='pearson', null_corr=None):
    """
    
    # data must be a matrix of shape num_vars X num_samples
    """
    if measure == 'pearson':
        compute_corr = pearson
    elif measure == 'distance':
        compute_corr = distance
    else:
        print("{} is not (yet) a supported correlation measure".format(measure))
        
    if null_corr is None:
        null_corr = compute_corr(data)

    # initialize aux vars used in loop
    p_values = np.zeros(null_corr.shape)
    num_loops = num_resamples if measure != 'distance' else int(np.ceil(num_resamples / 2))
    for _ in range(num_loops):
        perm_corr = compute_corr(data, perm=True)
        p_values += np.array(perm_corr>=null_corr, int)

    # trick for halfing num loops needed for num_resamples because of
    # distcov assymetry; only works if threshold is (nontrivially
    # above) 0
    p_values += p_values.T    
    p_values /= num_loops if measure != 'dcor' else 2 * num_loops

    return p_values, null_corr


def distance(data, perm=False):
    if not perm:
        with Pool() as pool:
            corr = pairwise(distcorr, data, pool=pool)
    else:
        data_permed = permute_within_rows(data)
        data_permed_2 = permute_within_rows(data)
            with Pool() as pool:
                corr = pairwise(distcorr, data_permed, data_permed_2, pool=pool)
    return corr


def pearson(data, num_resamples, measure='pearson', null_corr=None):
    if not perm:
        corr = np.corrcoef(data)
    else:
        data_permed = permute_within_rows(data)
        corr = np.corrcoef(data, data_permed)
    return corr


def permute_within_rows(x):
    # get random new index for col of each element
    col_idx = np.random.sample(x.shape).argsort(axis=1)

    # keep the row index the same
    row_idx = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T

    # apply the permutaton matrix to permute x
    return x[row_idx, col_idx]
