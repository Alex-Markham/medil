import numpy as np
# from networkx import from_numpy_matrix, Graph, draw
# from matplotlib.pyplot import show
# from networkx.drawing.nx_pydot import write_dot
# from graphviz import Digraph, render


# make these into static methods in the indep_test method?
def permute_within_columns(x):
    # get random new index for row of each element
    row_idx = np.random.sample(x.shape).argsort(axis=0)

    # keep the column index the same
    col_idx = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))

    # apply the permutaton matrix to permute x
    return x[row_idx, col_idx]


def permute_within_rows(x):
    # get random new index for col of each element
    col_idx = np.random.sample(x.shape).argsort(axis=1)

    # keep the row index the same
    row_idx = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T

    # apply the permutaton matrix to permute x
    return x[row_idx, col_idx]


# these go into test_medil.py
def gen_test_data(num_vars=5, num_samples=100):
    x = np.empty((num_vars, num_samples))#,dtype=')
    x[0, :] = np.arange(num_samples)#, dtype='float64')
    x[1, :] = np.arange(num_samples)*3#, dtype='float64') * 3
    x[2, :] = np.cos(np.arange(num_samples))#, dtype='float64'))
    x[3, :] = .7
    return x


test_graph_triangle = np.asarray([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1]])


test_graph_cm_am = []


# probably gets its own method? no, maybe its rather a property of the
# graph object returned by ECC finding? or is ECC finding just a sub
# func in the class/method (ahh, don't know termin) of the UDG, which
# could be a property of the data?
def display_graph(adjacency_matrix):
    graph = from_numpy_matrix(adjacency_matrix, create_using=Graph())
    draw(graph)
    show()

# def display_graph(adjacency_matrix, name):
#     path = '/home/alex_work/Documents/causal_concept_discovery/software/misc/'
#     graph = from_numpy_matrix(adjacency_matrix, create_using=DiGraph())
#     write_dot(graph, path + name + '.gv')
#     graph = Digraph('g', filename=path+name+'.gv')
#     graph.render
