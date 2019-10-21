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
        self.num_edges = 0


    def add_edges(self, edges):
        # doesn't behave well unless input is nparray;
        v_1s = edges[:, 0]
        v_2s = edges[:, 1]
        self.adj_matrix[v_1s, v_2s] = 1
        self.adj_matrix[v_2s, v_1s] = 1
        self.num_edges = self.adj_matrix.sum() // 2


    def rm_edges(self, edges):
        # doesn't behave well unless input is nparray;
        v_1s = edges[:, 0]
        v_2s = edges[:, 1]
        self.adj_matrix[v_1s, v_2s] = 0
        self.adj_matrix[v_2s, v_1s] = 0
        self.num_edges = self.adj_matrix.sum() // 2


# class minMCM(object):
# implement as a bigraph with biadjacency matrix with rows M and cols L


# class MCM(object):
# implement as large DAG adj matrix over L and M, or smaller bigraph for L-M connections and DAG adj matirx for L->L connections
