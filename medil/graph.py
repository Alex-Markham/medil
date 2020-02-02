"""Class definitions for various types and representations of graphs

The algorithms for finding the minMCM via ECC contain many algebraic
operations, so adjacency matrix representation (via NumPy) is most
covenient.

"""
import numpy as np


class UndirectedDependenceGraph(object):

    def __init__(self, adj_matrix):
        # doesn't behave well unless input is nparray;
        self.adj_matrix = adj_matrix
        self.num_vertices = len(adj_matrix)
        self.num_edges = adj_matrix.sum() // 2

    def add_edges(self, edges):
        v_1s = edges[:, 0]
        v_2s = edges[:, 1]
        self.adj_matrix[v_1s, v_2s] = 1
        self.adj_matrix[v_2s, v_1s] = 1
        self.num_edges = self.adj_matrix.sum() // 2

    def rm_edges(self, edges):
        v_1s = edges[:, 0]
        v_2s = edges[:, 1]
        self.adj_matrix[v_1s, v_2s] = 0
        self.adj_matrix[v_2s, v_1s] = 0
        self.num_edges = self.adj_matrix.sum() // 2

    def make_aux(self):
        # find neighbourhood for each vertex
        # each row corresponds to a unique edge
        max_num_edges = self.n_choose_2(self.num_vertices)
        self.common_neighbors= np.zeros((max_num_edges, self.num_vertices), int) # init

        # mapping of edges to unique row idx
        triu_idx = np.triu_indices(self.num_vertices, 1)
        nghbrhd_idx = np.zeros((self.num_vertices, self.num_vertices), int)
        nghbrhd_idx[triu_idx] = np.arange(max_num_edges)
        # nghbrhd_idx += nghbrhd_idx.T
        self.get_idx = lambda edge: nghbrhd_idx[edge[0], edge[1]]
        
        # reverse mapping
        # edges_idx = np.transpose(triu_idx)
        
        # compute actual neighborhood for each edge = (v_1, v_2)
        nbrs = lambda edge: np.logical_and(self.adj_matrix[edge[0]], self.adj_matrix[edge[1]])    
        
        extant_edges = np.transpose(np.triu(self.adj_matrix, 1).nonzero())
        self.extant_edges_idx = np.fromiter({self.get_idx(edge) for edge in extant_edges}, dtype=int)
        extant_nbrs = np.array([nbrs(edge) for edge in extant_edges])
        extant_nbrs_idx = np.array([self.get_idx(edge) for edge in extant_edges])
        
        self.common_neighbors[extant_nbrs_idx] = extant_nbrs
        
        # number of cliques for each node? assignments? if we set diag=0
        # num_cliques = common_neighbors.sum(0)
        
        # sum() of submatrix of graph containing exactly the rows/columns
        # corresponding to the nodes in common_neighbors(edge) using
        # logical indexing:
        
        # make mask to identify subgraph (closed common neighborhood of
        # nodes u, v in edge u,v)
        mask = lambda edge_idx: np.array(self.common_neighbors[edge_idx], dtype=bool)
        
        # make subgraph-adjacency matrix, and then subtract diag and
        # divide by two to get num edges in subgraph---same as sum() of
        # triu(subgraph-adjacency matrix) but probably a bit faster
        self.nbrhood = lambda edge_idx: self.adj_matrix[mask(edge_idx)][:, mask(edge_idx)]
        max_num_edges_in_nbrhood = lambda edge_idx: (self.nbrhood(edge_idx).sum() - mask(edge_idx).sum()) // 2
        
        self.nbrhood_edge_counts = np.array([max_num_edges_in_nbrhood(edge_idx) for edge_idx in np.arange(max_num_edges)])

        # important structs are:
        # self.common_neighbors 
        # self.nbrhood_edge_counts
        # # and fun is
        # self.nbrhood

    @staticmethod
    def n_choose_2(n):
        return n * (n - 1) // 2


class ReducedUndDepGraph(UndirectedDependenceGraph):
    pass



# class minMCM(object):
# implement as a bigraph with biadjacency matrix with rows M and cols L


# class MCM(object):
# implement as large DAG adj matrix over L and M, or smaller bigraph for L-M connections and DAG adj matirx for L->L connections

