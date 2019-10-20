"""Class definitions for various types and representations of graphs

The algorithms for finding the minMCM via ECC contain many algebraic
operations, so adjacency matrix representation (via NumPy) is most
covenient.

"""
import numpy as np


class UndirectedDependenceGraph(object):
    def __init__(self, num_vertices):
        # num_vertices is an int
        self.num_vertices = num_vertices
        self.adj_matrix = np.zeros((num_vertices, num_vertices), int)

        # auxiliary data structure to speed up computation
        self.num_edges = 0

        # reduced
        self.reduced = reduce_graph()

    def add_edges(self, edges):
        # doesn't behave well unless input is nparray;
        v_1s = edges[:, 0]
        v_2s = edges[:, 1]
        self.adj_matrix[v_1s, v_2s] = 1
        self.adj_matrix[v_2s, v_1s] = 1
        self.update_aux()


    def rm_edges(self, edges):
        # doesn't behave well unless input is nparray;
        v_1s = edges[:, 0]
        v_2s = edges[:, 1]
        self.adj_matrix[v_1s, v_2s] = 0
        self.adj_matrix[v_2s, v_1s] = 0
        self.update_aux()


    def update_aux(self):
        self.num_edges = self.adj_matrix.sum() // 2


    def reduce_graph(self):
        pass
