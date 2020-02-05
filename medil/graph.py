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

    def reduceable_copy(self):           # remove 'uncovered graph', since now we can just delet edges when they're covered, which is prob th e poist acutally of rule 3
        return ReduceableUndDepGraph(self)
    

class ReduceableUndDepGraph(UndirectedDependenceGraph):

    def __init__(self, udg):
        self.unreduced = udg
        self.adj_matrix = udg.adj_matrix.copy()
        self.num_vertices = udg.num_vertices
        self.num_edges = udg.num_edges

        # from auxilliary structure
        self.common_neighbors = udg.common_neighbors.copy()
        self.nbrhood_edge_counts = udg.nbrhood_edge_counts.copy()
        # and fun is
        self.nbrhood = udg.nbrhood  # need to fix this :/ gotta update if other stuff changes
        
        self.cover = None
        
    def reset(self):
        self.__init__(self.unreduced)
        
    def reduce(self, verbose=False):
        if verbose:
            print('\t\treducing:')
        self.reducing = True
        while self.reducing:
            self.reducing = False
            self.rule_1(self, verbose)
            self.rule_2(self, verbose)
            self.rule_3(self, verbose)

    def rule_1(self, verbose):
        # rule_1: Remove isolated vertices and vertices that are only
        # adjacent to covered edges

        isolated_verts = np.where(self.adj_matrix.sum(0)+self.adj_matrix.sum(1)==2)[0]
        if len(isolated_verts) > 0: # then Rule 1 is applied
            if verbose:
                print("\t\t\tapplying Rule 1...")
            
            self.adj_matrix[isolated_verts, isolated_verts] = 0

    def rule_2(self, verbose):
        # rule_2: If an uncovered edge {u,v} is contained in exactly
        # one maximal clique C, then add C to the solution, mark its
        # edges as covered, and decrease k by one
    
        # only check uncovered edges---may cause bugs?
        c_e_i = np.array([self.get_idx(x) for x in np.transpose(np.where(np.logical_and(self.adj_matrix==0, np.tri(self.num_vertices, k=-1).T)))], dtype=int)

        self.nbrhood_edge_counts[covered_edges_idx] = 0  # zeros out a row
        self.common_neighbors[covered_edges_idx] = 0  # zeros out a row
        
        # edges in at least 1 maximal clique
        at_least = self.nbrhood_edge_counts > 0
        
        # edges in at most 1 maximal clique
        at_most = (self.n_choose_2(self.common_neighbors.sum(1)) - self.nbrhood_edge_counts) == 0

        # cliques containing edges in exactly 1 maximal clique
        cliques = self.common_neighbors[at_least & at_most]

        if cliques.any():       # then apply Rule 2
            if verbose:
                print("\t\t\tapplying Rule 2...")
            cliques = np.unique(cliques, axis=0) # need to fix
            the_cover = cliques if the_cover is None else np.vstack((the_cover, cliques))
            uncovered_graph = cover_edges(uncovered_graph, cliques, verbose)  # replace with zeroing edges?
            counter -= 1
            continue            # start the reducee loop over so Rule
                                # 1 can 'clean up'


    def rule_3(self, verbose):
        # rule_3: Consider a vertex v that has at least one
        # guest. If inhabitants (of the neighborhood) occupy the
        # gates, then delete v. To reconstruct a solution for the
        # unreduced instance, add v to every clique containing a
        # guest of v. (prisoneers -> guests; dominate ->
        # occupy; exits -> gates; hierarchical social structures and
        # relations needn't be reproduced in mathematical
        # abstractions)

        exits = np.zeros((uncovered_graph.shape), dtype=bool)
            
        for vert, nbrhood in enumerate(uncovered_graph):
            if nbrhood[vert]==0: # then nbrhood is empty
                continue
            nbrs = np.flatnonzero(nbrhood)
            for nbr in nbrs:
                if (nbrhood - uncovered_graph[nbr] == -1).any():
                    exits[vert, nbr] = True
        # exits[i, j] == True iff j is an exit for i

        # guests[i, j] == True iff j is a guest of i
        guests = np.logical_and(~exits, uncovered_graph)

        applied_3 = False
        for pair in np.transpose(np.where(guests)):
            if the_cover is None:
                break
            guest_rooms_idx = np.transpose(np.where(the_cover[:, pair[1]]))
            
            if np.logical_not(the_cover[guest_rooms_idx, pair[1]]).any(): # then apply rule
                applied_3 = True
                if verbose:
                    print("\t\t\tapplying Rule 3...")
            # add host to all cliques containing guest
            the_cover[guest_rooms_idx, pair[1]] = 1
            uncovered_graph = cover_edges(uncovered_graph, the_cover, verbose)
        if applied_3:
            continue

    def choose_edge(self):    
        score = self.n_choose_2(self.common_neighbors.sum(1)) - self.nbrhood_edge_counts
        # score includes scores for non-existent edges, so have exclude those, otherwise could use .argmin()
        chosen_edge_idx = np.where(score==score[self.extant_edges_idx].min())[0][0]
        return chosen_edge_idx

    def remaining_uncovered(self, the_cover):
        if the_cover is None:
            return self.adj_mat
    
        # slightly more convenient representation and obviates need for deepcopy
        uncovered_graph = np.triu(graph_adj_mat)
    
        # change edges to 0 if they're covered
        for clique in the_cover:
            covered = clique.nonzero()[0]
        
            # trick for getting combinations from idx
            comb_idx = np.triu_indices(len(covered), 1)
        
            # actual combinations
            covered_row = covered[comb_idx[0]]
            covered_col = covered[comb_idx[1]]
            
            # cover edges
            uncovered_graph[covered_row, covered_col] = 0
        return uncovered_graph

    if verbose:
        print("\t\t\t{} uncovered edges remaining".format(uncovered_graph.sum()))
    return np.triu(uncovered_graph, 1) + uncovered_graph.T
        
    @staticmethod
    def max_cliques(nbrhood):
        # pieced together from nx.from_numpy_array and nx.find_cliques
        # nx.find_cliques is output sensitive :)

        # convert adjacency matrix to nx style graph
    
        # adapted from nx.find_cliques to find max cliques
        if len(nbrhood) == 0:
            return

        adj = {u: {v for v in np.nonzero(nbrhood[u])[0] if v != u} for u in range(len(nbrhood))}
        Q = [None]

        subg = set(range(len(nbrhood)))
        cand = set(range(len(nbrhood)))
        u = max(subg, key=lambda u: len(cand & adj[u]))
        ext_u = cand - adj[u]
        stack = []

        try:
            while True:
                if ext_u:
                    q = ext_u.pop()
                    cand.remove(q)
                    Q[-1] = q
                    adj_q = adj[q]
                    subg_q = subg & adj_q
                    if not subg_q:
                        yield Q[:]
                    else:
                        cand_q = cand & adj_q
                        if cand_q:
                            stack.append((subg, cand, ext_u))
                            Q.append(None)
                            subg = subg_q
                            cand = cand_q
                            u = max(subg, key=lambda u: len(cand & adj[u]))
                            ext_u = cand - adj[u]
                else:
                    Q.pop()
                    subg, cand, ext_u = stack.pop()
        except IndexError:
            pass    
        # note: max_cliques is a generator, so it's consumed after being
        # looped through once





# class minMCM(object):
# implement as a bigraph with biadjacency matrix with rows M and cols L


# class MCM(object):
# implement as large DAG adj matrix over L and M, or smaller bigraph for L-M connections and DAG adj matirx for L->L connections

