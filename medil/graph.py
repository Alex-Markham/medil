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
        common_neighbors= np.zeros((num_edges, num_vertices), int) # init
        
        # mapping of edges to unique row idx
        nghbrhd_idx = np.zeros((num_vertices, num_vertices), int)
        nghbrhd_idx[triu_idx] = np.arange(num_edges)
        # nghbrhd_idx += nghbrhd_idx.T
        get_idx = lambda edge: nghbrhd_idx[edge[0], edge[1]]
        
        # reverse mapping
        # edges_idx = np.transpose(triu_idx)
        
        # compute actual neighborhood for each edge = (v_1, v_2)
        nbrs = lambda edge: np.logical_and(graph[edge[0]], graph[edge[1]])    
        
        extant_edges = np.transpose(np.triu(graph, 1).nonzero())
        extant_nbrs = np.array([nbrs(edge) for edge in extant_edges])
        extant_nbrs_idx = [get_idx(edge) for edge in extant_edges]
        
        self.common_neighbors[extant_nbrs_idx] = extant_nbrs
        
        # number of cliques for each node? assignments? if we set diag=0
        # num_cliques = common_neighbors.sum(0)
        
        # sum() of submatrix of graph containing exactly the rows/columns
        # corresponding to the nodes in common_neighbors(edge) using
        # logical indexing:
        
        # make mask to identify subgraph (closed common neighborhood of
        # nodes u, v in edge u,v)
        mask = lambda edge_idx: np.array(common_neighbors[edge_idx], dtype=bool)
        
        # make subgraph-adjacency matrix, and then subtract diag and
        # divide by two to get num edges in subgraph---same as sum() of
        # triu(subgraph-adjacency matrix) but probably a bit faster
        self.nbrhood = lambda edge_idx: graph[mask(edge_idx)][:, mask(edge_idx)]
        num_edges_in_nbrhood = lambda edge_idx: (self.nbrhood(edge_idx).sum() - mask(edge_idx).sum()) // 2
        
        nbrhood_edge_counts = np.array([num_edges_in_nbrhood(edge_idx) for edge_idx in np.arange(num_edges)])

        # important structs are:
        # self.common_neighbors 
        # self.nbrhood_edge_counts
        # # and fun is
        # self.nbrhood
        
    


class ReducedUndDepGraph(UndirectedDependenceGraph):
    pass



# class minMCM(object):
# implement as a bigraph with biadjacency matrix with rows M and cols L


# class MCM(object):
# implement as large DAG adj matrix over L and M, or smaller bigraph for L-M connections and DAG adj matirx for L->L connections

