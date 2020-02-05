"""Class definitions for various types and representations of graphs

The algorithms for finding the minMCM via ECC contain many algebraic
operations, so adjacency matrix representation (via NumPy) is most
covenient.

"""
import numpy as np


class UndirectedDependenceGraph(object):

    def __init__(self, adj_matrix, verbose=False):
        # doesn't behave well unless input is nparray;
        self.adj_matrix = adj_matrix
        self.num_vertices = len(adj_matrix)
        self.num_edges = adj_matrix.sum() // 2
        self.verbose = verbose

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

    def reducible_copy(self):
        return ReducibleUndDepGraph(self)
    

class ReducibleUndDepGraph(UndirectedDependenceGraph):

    def __init__(self, udg, verbose=False):
        self.unreduced = udg
        self.adj_matrix = udg.adj_matrix.copy()
        self.num_vertices = udg.num_vertices
        self.num_edges = udg.num_edges

        self.the_cover = None
        self.verbose = verbose
        
        # from auxilliary structure
        self.get_idx = udg.get_idx

        # need to update these all when self.cover_edges() is called?
        self.common_neighbors = udg.common_neighbors.copy()
        self.nbrhood_edge_counts = udg.nbrhood_edge_counts.copy()
        # and fun is
        self.nbrhood = udg.nbrhood  # need to fix this :/ gotta update if other stuff changes
        
        extant_edges = np.transpose(np.triu(self.adj_matrix, 1).nonzero())
        self.extant_edges_idx = np.fromiter({self.get_idx(edge) for edge in extant_edges}, dtype=int)
        
    def reset(self):
        self.__init__(self.unreduced)
        
    def reduzieren(self, k_num_cliques):
        if self.verbose:
            print('\t\treducing:')
        self.k_num_cliques = k_num_cliques
        self.reducing = True
        while self.reducing:
            self.reducing = False
            self.rule_1()
            self.rule_2()
            self.rule_3()

    def rule_1(self):
        # rule_1: Remove isolated vertices and vertices that are only
        # adjacent to covered edges

        isolated_verts = np.where(self.adj_matrix.sum(0)+self.adj_matrix.sum(1)==2)[0]
        if len(isolated_verts) > 0: # then Rule 1 is applied
            if self.verbose:
                print("\t\t\tapplying Rule 1...")
            
            self.adj_matrix[isolated_verts, isolated_verts] = 0

    def rule_2(self):
        # rule_2: If an uncovered edge {u,v} is contained in exactly
        # one maximal clique C, then add C to the solution, mark its
        # edges as covered, and decrease k by one
    
        # only check uncovered edges---may cause bugs?
        covered_edges_idx = np.array([self.get_idx(x) for x in np.transpose(np.where(np.logical_and(self.adj_matrix==0, np.tri(self.num_vertices, k=-1).T)))], dtype=int)

        self.nbrhood_edge_counts[covered_edges_idx] = 0  # zeros out a row
        self.common_neighbors[covered_edges_idx] = 0  # zeros out a row
        
        # edges in at least 1 maximal clique
        at_least = self.nbrhood_edge_counts > 0
        
        # edges in at most 1 maximal clique
        at_most = (self.n_choose_2(self.common_neighbors.sum(1)) - self.nbrhood_edge_counts) == 0

        # cliques containing edges in exactly 1 maximal clique
        cliques = self.common_neighbors[at_least & at_most]

        if cliques.any():       # then apply Rule 2
            if self.verbose:
                print("\t\t\tapplying Rule 2...")
            cliques = np.unique(cliques, axis=0) # need to fix? just not as efficient as possible
            self.the_cover = cliques if self.the_cover is None else np.vstack((self.the_cover, cliques))
            self.cover_edges()
            self.k_num_cliques -= len(cliques)
            return             # or rather self.rule_1()?y
        # start the reducee loop over so Rule
                                # 1 can 'clean up'


    def rule_3(self):
        # rule_3: Consider a vertex v that has at least one
        # guest. If inhabitants (of the neighborhood) occupy the
        # gates, then delete v. To reconstruct a solution for the
        # unreduced instance, add v to every clique containing a
        # guest of v. (prisoneers -> guests; dominate ->
        # occupy; exits -> gates; hierarchical social structures and
        # relations needn't be reproduced in mathematical
        # abstractions)

        exits = np.zeros((self.adj_matrix.shape), dtype=bool)
            
        for vert, nbrhood in enumerate(self.adj_matrix):
            if nbrhood[vert]==0: # then nbrhood is empty
                continue
            nbrs = np.flatnonzero(nbrhood)
            for nbr in nbrs:
                if (nbrhood - self.adj_matrix[nbr] == -1).any():
                    exits[vert, nbr] = True
        # exits[i, j] == True iff j is an exit for i

        # guests[i, j] == True iff j is a guest of i
        guests = np.logical_and(~exits, self.adj_matrix)

        applied_3 = False
        for pair in np.transpose(np.where(guests)):
            if self.the_cover is None:
                break
            guest_rooms_idx = np.transpose(np.where(self.the_cover[:, pair[1]]))
            
            if np.logical_not(self.the_cover[guest_rooms_idx, pair[1]]).any(): # then apply rule
                applied_3 = True
                if self.verbose:
                    print("\t\t\tapplying Rule 3...")
            # add host to all cliques containing guest
            self.the_cover[guest_rooms_idx, pair[1]] = 1
            self.cover_edges()
        if applied_3:
            return               #  need to start loop over?

    def choose_edge(self):    
        score = self.n_choose_2(self.common_neighbors.sum(1)) - self.nbrhood_edge_counts
        # score includes scores for non-existent edges, so have exclude those, otherwise could use .argmin()
        chosen_edge_idx = np.where(score==score[self.extant_edges_idx].min())[0][0]
        return chosen_edge_idx

    def cover_edges(self):
        # always call after updating the cover
        if self.the_cover is None:
            return self.adj_matrix
    
        # change edges to 0 if they're covered
        for clique in self.the_cover:
            covered = clique.nonzero()[0]
        
            # trick for getting combinations from idx
            comb_idx = np.triu_indices(len(covered), 1)
        
            # actual pairwise combinations; ie all edges (v_i, v_j) covered by the clique
            covered_edges = np.empty((len(comb_idx[0]), 2), int)
            covered_edges[:, 0] = covered[comb_idx[0]]
            covered_edges[:, 1] = covered[comb_idx[1]]
            
            # cover (remove from reduced_graph) edges
            self.rm_edges(covered_edges)

        if self.verbose:
            print("\t\t\t{} uncovered edges remaining".format(self.num_edges))

        # now here do all the updates to nbrs, extant edges, etc.



# class minMCM(object):
# implement as a bigraph with biadjacency matrix with rows M and cols L


# class MCM(object):
# implement as large DAG adj matrix over L and M, or smaller bigraph for L-M connections and DAG adj matirx for L->L connections

