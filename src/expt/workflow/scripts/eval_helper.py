import numpy as np
from dcor import distance_correlation
from scipy.optimize import linear_sum_assignment


def mcc(true, est):
    true = (true - true.mean(0)) / (true.std(0) + 1e-8)
    est = (est - est.mean(0)) / (est.std(0) + 1e-8)

    corr = np.abs(true.T @ est) / true.shape[0]
    row_ind, col_ind = linear_sum_assignment(corr, maximize=True)
    return corr[row_ind, col_ind].mean()


def dcor(true, est):
    return distance_correlation(true, est)


def mse(true, est):
    return np.mean(np.square(true - est))
