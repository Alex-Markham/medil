import numpy as np


def get_depends(file, alpha=.05):
    p = np.load(file)['p']
    # generate dependence matrix D: with d_ij == (0)1 means x_i and
    # x_j are (in)depependent
    return np.triu(p <= alpha, 1)


def matrix_to_edges(depends):
    edges = np.transpose(np.where(depends))
    edges = ['v' + str(v_1) + ' v' + str(v_2) for v_1, v_2 in edges]
    return '\n'.join(edges)


d_m = get_depends('BIG5_perm.npz')
edges = matrix_to_edges(d_m)

with open("edges.txt", "w") as text_file:
    text_file.write(edges)
