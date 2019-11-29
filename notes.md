  only:
    changes:
    - docs/*
  - master

# use np.ravel_index #

[for conda package](https://stackoverflow.com/questions/49474575/how-to-install-my-own-python-module-package-via-conda-and-watch-its-changes)

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
  * [pipelines for maintaining code integrity](https://www.patricksoftwareblog.com/setting-up-gitlab-ci-for-a-python-application/)
  
Readthedocs website for medil.causal.dev:
  * [adding custom domain](https://docs.readthedocs.io/en/stable/custom_domains.html)
  * [specifying canonical URL](https://docs.readthedocs.io/en/stable/guides/canonical.html)
  * [self hosting docs with subtree](https://stackoverflow.com/questions/45565464/git-push-subfolder-to-different-repository)

Check out [this](http://signal.ee.psu.edu/mrf.pdf) slidedeck for a nice summary of MRF and related graph theory concepts

See [this](https://en.wikipedia.org/wiki/Markov_random_field) as well as factor graphs and clique factorization

[testing in python](https://realpython.com/python-testing/)
[more testing](https://docs.python-guide.org/writing/tests/)

What I'm doing is called [Monte Carlo Testin](https://en.wikipedia.org/wiki/Resampling_(statistics)#Monte_Carlo_testing)

Can probably make inverse ECC generator that takes a given (randmoly generated) covering and outputs a (random?) graph for which it is the min ECC to do large-scale testing? But there can be multiple min ECCs :/
*Definitely* can add am/cm diff test graph




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

dag and print:
```python
import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from string import ascii_uppercase as Alpha


def plot_bar_chart(ax, fields, purity_scores, total_cons):
    # sort?
    ax.bar(fields, purity_scores)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.text(0.1, 0.9, 'test', ha='center', va='center',
            transform=ax.transAxes)


def plot_graph(adjacency_matrix, fields, name):
    plt.figure(num=1, figsize=(20, 20))
    graph = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.DiGraph())
    labels = {idx: "C " + str(idx) if idx < num_concepts else
              fields[idx - num_concepts] for idx in
              range(num_total)}
    colors = ['r' if idx < num_concepts else 'b' for idx in
              range(num_total)]
    nx.draw(graph, labels=labels, with_labels=True,
            node_color=colors, node_size=700)
    plt.savefig(name + '.png')
#    plt.show()
    plt.close()
    return graph, num_concepts  # , adjacency_matrix


def make_adj_matrix(all_concepts):
        # make adjacency matrix from c_graph
    num_concepts, num_observed = all_concepts.shape
    num_total = sum(all_concepts.shape)

    adjacency_matrix = np.zeros((num_total, num_total))
    adjacency_matrix[:num_concepts, -num_observed:] = all_concepts
    return adjacency_matrix


def make_concept_graphs(depends):
    # :param depends: dependence matrix D: with d_ij == (0)1 means x_i
    # and x_j are (in)depependent

    num_vars = depends.shape[0]

    c_graphs = dict()  # init dict containg concept graph for each var
    # loop through vars to make each concept graph
    for root_var in range(num_vars):
        # initialize concept graph
        concept_graph = np.zeros((1, num_vars), dtype='bool')
        concept_graph[0, root_var] = True

        # get array of dependent vars (excluding self-dependence)
        dep_vars = np.array(depends[root_var])
        dep_vars[root_var] = False
        dep_vars = np.flatnonzero(dep_vars)

        # loop through dep_vars, adding each to concept graph
        for dep_var in dep_vars:
            for concept in concept_graph:
                # add to concept iff it's dep with all other vars
                if not depends[dep_var, np.flatnonzero(concept)].all():
                    continue
                concept[dep_var] = True
                # if dep_var indep of all concepts, add new concept
            if not concept_graph[:, dep_var].any():
                concept_graph = add_new_concept(root_var,
                                                concept_graph,
                                                dep_var, dep_vars,
                                                depends)
        c_graphs[root_var] = np.asarray(concept_graph, dtype='int')

    all_concepts = [row for idx in c_graphs for row in
                    c_graphs[idx]]
    all_concepts = np.unique(all_concepts, axis=0)

    return c_graphs, all_concepts


def get_depends(x=None, name=None, alpha=.05, file=None):
    if file is not None:
        p = np.load(file)['p']
    else:
        rho, p = rhoperm(x, name, 1000)

    # generate dependence matrix D: with d_ij == (0)1 means x_i and
    # x_j are (in)depependent
    return p <= alpha  # , rho, p


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


def permute_within_columns(x):
    # get random new index for row of each element
    row_idx = np.random.sample(x.shape).argsort(axis=0)

    # keep the column index the same
    col_idx = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))

    # apply the permutaton matrix to permute x
    return x[row_idx, col_idx]


def add_new_concept(var_eff, c_graph, var_caus, dep_vars, dependent):
    num_vars = c_graph.shape[1]

    new_concept = np.zeros((1, num_vars), dtype='bool')
    # make concept dep with var_eff and var_caus
    new_concept[0, [var_eff, var_caus]] = [True, True]

    # loop through previous var_causes to add to new concept; only add
    # if prev_var_caus is dep with all other vars in concept
    for prev_var_caus in dep_vars[dep_vars < var_caus]:
        if dependent[prev_var_caus, np.flatnonzero(new_concept)].all():
            new_concept[0, prev_var_caus] = True
    return np.append(c_graph, new_concept, axis=0)

######################################################################
# personality tests data

# PF16
# pf_data = np.loadtxt('16PF/data.csv', delimiter='\t',
#                      skiprows=1, usecols=np.arange(1, 163))
# with open('16PF/data.csv') as file:
#     pf_fields = np.asarray(file.readline().split('"\t"')[:163])

# BIG5
b5_data = np.loadtxt('BIG5/data.csv', delimiter='\t',
                     skiprows=1, usecols=np.arange(7, 57))
with open('BIG5/data.csv') as file:
    b5_fields = np.asarray(file.readline().split('\t')[7:57])
b5_fields[-1] = b5_fields[-1][:-1]

######################################################################


def dag_n_print(data, fields, name):
    # plot_graph
    depends = get_depends(data, file=name + '_perm.npz')
    c_graph = find_concept_graphs(depends)
    graph, num_total_concepts = plot_graph(c_graph, fields, name)

    # # for plotting
    # total_cons_dict[key] = num_total_concepts
    # purity_dict[key] = np.array([c_graph[g].shape[0] for g in
    #                                  c_graph])

    # printout
    print(f"number total concepts for {name} graph: " +
          str(num_total_concepts))
    with open(name + '.csv', 'w') as file:
        # file.write(f"number total concepts for {name} graph: " +
        #            str(num_total_concepts))
        for i in range(len(fields)):
            file.write(f"\n{fields[i]}, {c_graph[i].shape[0]}, " +
                       f"{sum(depends)[i] - 1}")

    return c_graph


# c_graph = dag_n_print(b5_data, b5_fields, 'BIG5')
# dag_n_print(pf_data, pf_fields, '16PF')

######################################################################
# diff traits

b5_dict = {'E5': np.arange(10),
           'N5': np.arange(10, 20),
           'A5': np.arange(20, 30),
           'C5': np.arange(30, 40),
           'O5': np.arange(40, 50)}

# pf_dict = {Alpha[i]: np.arange(i * 10 + 3, (i + 1) * 10 + 3)
#            for i in np.arange(2, 16)}
# pf_dict['A'] = np.arange(10)
# pf_dict['B'] = np.arange(10, 23)


def dag_n_print(idx_dict, all_data, all_fields):
    field_dict = dict()
    purity_dict = dict()
    total_cons_dict = dict()
    for key in idx_dict:
        idx = idx_dict[key]
        data = all_data[:, idx]
        fields = all_fields[idx]
        field_dict[key] = fields

        # plot_graph
        depends = get_depends(data)
        all_concepts = make_concepts(depends)
        graph, num_total_concepts = plot_graph(all_concepts, fields,
                                               key)

        # for plotting
        total_cons_dict[key] = num_total_concepts
        purity_dict[key] = np.array([c_graph[g].shape[0] for g in
                                     c_graph])

        # printout
        print(f"number total concepts for {key} graph: " +
              str(num_total_concepts))
        with open(key + '.csv', 'w') as file:
            for i in range(len(fields)):
                file.write(f"\n{fields[i]}, {c_graph[i].shape[0]}, " +
                           f"{sum(depends)[i]}")
    return field_dict, purity_dict, total_cons_dict


# fd, pd, tcd = dag_n_print(b5_dict, b5_data, b5_fields)

######################################################################

# example_depends = np.ones((5, 5))
# example_depends[[0, 1, 2, 2, 3, 3, 4, 4], [1, 0, 3, 4, 2, 4, 2, 3]] = 0
# c_graph = find_concept_graphs(example_depends)
# plot_graph(c_graph, np.asarray(['1', '2', '3', '4', '5']), 'example')

# example2_depends = np.ones((6, 6))
# example2_depends[[0, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 3, 4, 2, 4, 2, 3,
#                                                4]] = 0
# c_graph2 = find_concept_graphs(example2_depends)
# plot_graph(c_graph2, np.asarray(['1', '2', '3', '4', '5', '6']), 'example2')
```
