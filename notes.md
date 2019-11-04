Pushing new release:
```bash
git push origin develop
git tag -a v0.X.0 -m "Releasing version 0.X.0"
git push origin v0.X.0
git push origin develop:master
```

Structuring the Project (in descending order of importance/usefulness/detail):

  * [seems to say everything, but is a lot of reading](https://docs.python-guide.org/writing/structure/)
  * [extensive---probably useful for a while](https://python-packaging.readthedocs.io/en/latest/minimal.html)
  * [short---covered in other refs---think i've already done it all]( https://able.bio/SamDev14/how-to-structure-a-python-project--685o1o6)

Readthedocs website for medil.causal.dev:
  * [adding custom domain](https://docs.readthedocs.io/en/stable/custom_domains.html)
  * [specifying canonical URL](https://docs.readthedocs.io/en/stable/guides/canonical.html)

Check out [this](http://signal.ee.psu.edu/mrf.pdf) slidedeck for a nice summary of MRF and related graph theory concepts

See [this](https://en.wikipedia.org/wiki/Markov_random_field) as well as factor graphs and clique factorization

Undirected dependency graph is complement of Markov Random Field with global Markov property? And so cliques of UDG would be independent sets---ecc of UDG corresponds to sufficient factorization of MRF into independent sets

[testing in python](https://realpython.com/python-testing/)
[more testing](https://docs.python-guide.org/writing/tests/)


Code:
  * could make a load module containing funcs such as from\_dict and from\_edges and from\_adj\_matrix etc, but not a big priority---network does it all well already, and medil really just needs to work with the output from the perm test methods
  * maybe change UndirectedDependenceGraph object init argument to be a adj matrix instead?

test results showing speed differences:
```python
import dcor
import numpy as np
import time
import multiprocessing
import numba
from utils import permute_within_columns, gen_test_data

x = gen_test_data()

# num loops for time
number = 1000

# pool = multiprocessing.Pool()
# # first function test    
# start = time.time()
# for _ in range(number):
#     x_p = permute_within_columns(x.T).T
#     x_pp = permute_within_columns(x.T).T
#     ps = dcor.pairwise(dcor.distance_covariance, x_pp, x_p, pool=pool)
# end = time.time()
# time1 = end - start
# print("\n time for function one: %.5f" % time1)
# 40 secs (so 20)

# start2 = time.time()
# # for _ in range(number):
# #     x_p = permute_within_columns(x.T).T
# #     ps = dcor.pairwise(dcor.distance_covariance, x_p)
# end2 = time.time()
# time2 = end2 - start2
# print("\n time for function two: %.5f" % time2)
# 40 sects

# they take basically the same amount of time, but the second test
# gives us 2 chances to update p value, so overall time is cut in half
# by using it
# ------------------------------------------------------------------------------------------------


# start3 = time.time() 
p_val_matrix = np.empty((num_vars, num_vars))
power_matrix = np.empty((num_vars, num_vars))

# for row, col in np.transpose(np.triu_indices(num_vars, 1)):
#     # get samples for each var
#     v_1 = x[row, :]
#     v_2 = x[col, :]

#     p_val, power = dcor.independence.distance_covariance_test(v_1, v_2, num_resamples=number)

#     p_val_matrix[row, col] = p_val
#     power_matrix[row, col] = power

# end3 = time.time()
# time3 = end3 - start3
# print("\n time for function three: %.5f", % time3)
# 80 secs

# start4 = time.time()
# @numba.jit(nopython=False, parallel=True)
# def perm_test(x, iterations):
#     for row, col in iterations:
#     # get samples for each var
#         v_1 = x[row, :]
#         v_2 = x[col, :]
        
#         results =  dcor.independence.distance_covariance_test(v_1.T, v_2.T, num_resamples=number)

#         p_val, power = results
        
#         p_val_matrix[row, col] = p_val
#         power_matrix[row, col] = power

#     return p_val_matrix, power_matrix

# iterations = np.transpose(np.triu_indices(num_vars, 1))
# p_val_matrix, power_matrix = perm_test(x, iterations)

# end4 = time.time()
# time4 = end4 - start4
# print("\n time for function four: %.5f" % time4)
# 80 secs (but may not have actually parallelized correctly

# as for num_samples = 1000, function 4 was (4x) faster than function 3/4
# for 100 samples, function 3/4 was (5x) faster
# -----------------------------------------------------------------------------------------------------------------

# # @jit
# def perm_test(data, num_vars=None, num_samps=None, num_resamps=100, measure='dcorr'):
#     # cols are vars and rows are samples
#     if num_vars is None:
#         num_vars = data.shape[1]

#     if num_samps is None:
#         num_samps = data.shape[0]

#     # get indices for each of the (num_vars choose 2) pairs
#     idx = np.tri(num_vars, k=-1) 

#     # init empty matrices to keep track of p_values and test power
#     p_val_matrix = np.empty((num_vars, num_vars))
#     power_matrix = np.empty((num_vars, num_vars))

#     # loop through each pair
#     for row, col in np.transpose(np.triu_indices(num_vars, 1)):
#         # get samples for each var
#         v_1 = data[:num_samps, row]
#         v_2 = data[:num_samps, col]

#         if measure == 'dcorr':
#             # run permutation test on the pair
#             # 10 num_resamp took 45 secs, 100 took 350, 1000 crashed memory
#             p_val, power = dct(v_1, v_2, num_resamples=num_resamps)
            
#             # record results
#             p_val_matrix[row, col] = p_val
#             power_matrix[row, col] = power

#         # else:
#         #     p_val = distcorr(v_1, v_2, num_runs=num_resamps)

#         else:
#             p_val = distCorr(v_1, v_2)
            

#     return p_val_matrix, power_matrix if measure == 'dcorr' else p_val


# # b5_data = np.loadtxt('data.csv', delimiter='\t',
# #                      skiprows=1, usecols=np.arange(7, 57))
# # # get field names
# # with open('data.csv') as file:
# #     b5_fields = np.asarray(file.readline().split('\t')[7:57])
# # b5_fields[-1] = b5_fields[-1][:-1]

# b5_data = np.load('b5_data.npy', mmap_mode='r')

# start = time.time()
# ps = perm_test(b5_data, num_vars=5, num_samps=500, num_resamps=100)# , measure='distCorr')
# end = time.time()
# print('time:', end - start)

# note: seems to scale linearly with num_vars and num_resamps but exponentially with num_samps
# dcorr: num_vars=2 and num_samps=1000 and num_resamps=1000 takes 8 secs
# ------ so all 19000 samps would take about 8 * 400 * 1225 secs = 50  days

# distcorr: takes 36 secs, so 4.5 times as long
```

alternative, slower distcorr implementation:
```python

from scipy.spatial.distance import pdist, squareform
import numpy as np
import copy


def distcorr(Xval, Yval, pval=True, num_runs=500):
    """ Compute the distance correlation function, returning the p-value.
    Based on Satra/distcorr.py (gist aa3d19a12b74e9ab7941)
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    (0.76267624241686671, 0.404)
    """
    X = np.atleast_1d(Xval)
    Y = np.atleast_1d(Yval)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    if pval:
        greater = 0
        for i in range(num_runs):
            Y_r = copy.copy(Yval)
            np.random.shuffle(Y_r)
            if distcorr(Xval, Y_r, pval=False) >= dcor:
                greater += 1
        return (dcor, greater / float(num_runs))
    else:
        return dcor
```
