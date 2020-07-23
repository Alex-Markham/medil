import numpy as np
from dcor import pairwise, distance_correlation as distcorr
from multiprocessing import Pool


def dependencies(null_hyp, iota, p_values, alpha):
    null_indep = null_hyp <= iota
    accept_null = p_values >= alpha
    independencies = null_indep & accept_null
    return ~independencies     # dependencies


def hypothesis_test(data, num_resamples, null_corr=None):
    # data must be a matrix of shape num_vars X num_samples
    
    if null_corr is None:
        with Pool() as pool:
            null_corr = pairwise(distcorr, data, pool=pool)
        # null_cov[np.isnan(null_cov)] = 0 # cov with uniform is nan?
        # np.seterr(over='ignore')
        # fixed it---just a problem with the type of my test data

    # initialize aux vars used in loop
    p_values = np.zeros(null_corr.shape)
    num_loops = int(np.ceil(num_resamples / 2))
    for _ in range(num_loops):
        x_1 = permute_within_rows(data)
        x_2 = permute_within_rows(data)

        with Pool() as pool:
            perm_corr = pairwise(distcorr, x_1, x_2, pool=pool)
        p_values += np.array(perm_corr>=null_corr, int)

    # trick for halfing num loops needed for num_resamples because of
    # distcov assymetry; only works if threshold is (nontrivially
    # above) 0
    p_values += p_values.T    
    p_values /= 2 * num_loops

    return p_values, null_corr


def rhoperm(x, name=None, num_perms=1000):
    '''Computes Pearson's correlation coefficient with p-values.

    rho, p = rhoperm(x, num_perms=1000)

    Input variable x has to be of dimension [observations X
    variables]. The p-values are computed by a permutation test.

    '''

    num_samps, num_vars = x.shape

    # Correlation matrix
    rho = np.corrcoef(x, rowvar=False)

    # permutation test for p-values
    p = np.zeros([num_vars, num_vars])
    alt_rho_idx = np.triu_indices(2 * num_vars, num_vars + 1)
    rho_idx = np.triu_indices(num_vars, 1)
    for perm in range(num_perms):
        x_perm = permute_within_columns(x)
        alt_rho = np.corrcoef(x, x_perm, rowvar=False)
        p[rho_idx] += abs(alt_rho[alt_rho_idx]) > abs(rho[rho_idx])
    p = (p + p.T) / num_perms

    if name is not None:
        np.savez(name + '_perm', rho=rho, p=p)
    return rho, p


def permute_within_rows(x):
    # get random new index for col of each element
    col_idx = np.random.sample(x.shape).argsort(axis=1)

    # keep the row index the same
    row_idx = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T

    # apply the permutaton matrix to permute x
    return x[row_idx, col_idx]
