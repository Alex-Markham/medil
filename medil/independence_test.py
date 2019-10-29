import numpy as np
from dcor import pairwise, distance_covariance as distcov
from multiprocessing import Pool
from .utils import permute_within_rows


def hypothesis_test(data, num_resamples, null_cov=None):
    # data must be a matrix of shape num_vars X num_samples
    
    if null_cov is None:
        with Pool() as pool:
            null_cov = pairwise(distcov, data, pool=pool)
        # null_cov[np.isnan(null_cov)] = 0 # cov with uniform is nan?
        # np.seterr(over='ignore')
        # fixed it---just a problem with the type of my test data

    # initialize aux vars used in loop
    p_values = np.zeros(null_cov.shape)
    num_loops = int(np.ceil(num_resamples / 2))
    for _ in range(num_loops):
        x_1 = permute_within_rows(data)
        x_2 = permute_within_rows(data)

        with Pool() as pool:
            perm_cov = pairwise(distcov, x_1, x_2, pool=pool)
        p_values += np.array(perm_cov>=null_cov, int)

    # trick for halfing num loops needed for num_resamples because of
    # distcov assymetry; only works if threshold is (nontrivially
    # above) 0
    p_values += p_values.T    
    p_values /= 2 * num_loops

    return p_values, null_cov


def dependencies(null_hyp, iota, p_values, alpha):
    null_indep = null_hyp <= iota
    accept_null = p_values >= alpha
    independencies = null_indep & accept_null
    return ~independencies     # dependencies
