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
        # doesn't behave well unless input is nparray; need to check
        # if edge is added twice for sake of num_edges?
        v_1s = edges[:, 0]
        v_2s = edges[:, 1]
        self.adj_matrix[v_1s, v_2s] = 1
        self.adj_matrix[v_2s, v_1s] = 1
        self.num_edges += len(edges)
