"""Various types and representations of graphs."""
import numpy as np


class UndirectedDependenceGraph(object):
    r"""Adjacency matrix representation using a 2d numpy array.

    Upon initialization, this class is fairly standard implementation
    of an undirected graph. However, upon calling the :meth:`make_aux`
    method, an auxilliary data structure in the form of several new
    attributes is created, which are used by
    :meth:`medil.ecc_algorithms.find_clique_min_cover` according to
    the algorithm in :cite:`Gramm_2009`.

    Attributes
    ----------


    Notes
    -----
    The algorithms for finding the minMCM via ECC contain many
    algebraic operations, so adjacency matrix representation (via
    NumPy) is most covenient.

    """

    def __init__(self, adj_matrix, verbose=False):
        # doesn't behave well unless input is nparray
        self.adj_matrix = adj_matrix
        self.num_vertices = np.trace(adj_matrix)
        self.max_num_verts = len(adj_matrix)
        self.num_edges = np.triu(adj_matrix, 1).sum()
        self.verbose = verbose

    def add_edges(self, edges):
        v_1s = edges[:, 0]
        v_2s = edges[:, 1]
        self.adj_matrix[v_1s, v_2s] = 1
        self.adj_matrix[v_2s, v_1s] = 1
        self.num_edges = np.triu(self.adj_matrix, 1).sum()

    def rm_edges(self, edges):
        v_1s = edges[:, 0]
        v_2s = edges[:, 1]
        self.adj_matrix[v_1s, v_2s] = 0
        self.adj_matrix[v_2s, v_1s] = 0
        self.num_edges = np.triu(self.adj_matrix, 1).sum()

    def make_aux(self):
        # this makes the auxilliary structure described in INITIALIZATION in the paper

        # find neighbourhood for each vertex
        # each row corresponds to a unique edge
        max_num_edges = self.n_choose_2(self.max_num_verts)
        self.common_neighbors = np.zeros(
            (max_num_edges, self.max_num_verts), int
        )  # init

        # mapping of edges to unique row idx
        triu_idx = np.triu_indices(self.max_num_verts, 1)
        nghbrhd_idx = np.zeros((self.max_num_verts, self.max_num_verts), int)
        nghbrhd_idx[triu_idx] = np.arange(max_num_edges)
        # nghbrhd_idx += nghbrhd_idx.T
        self.get_idx = lambda edge: nghbrhd_idx[edge[0], edge[1]]

        # reverse mapping
        u, v = np.where(np.triu(np.ones_like(self.adj_matrix), 1))
        self.get_edge = lambda idx: (u[idx], v[idx])

        # compute actual neighborhood for each edge = (v_1, v_2)
        self.nbrs = lambda edge: np.logical_and(
            self.adj_matrix[edge[0]], self.adj_matrix[edge[1]]
        )

        extant_edges = np.transpose(np.triu(self.adj_matrix, 1).nonzero())
        self.extant_edges_idx = np.fromiter(
            {self.get_idx(edge) for edge in extant_edges}, dtype=int
        )
        extant_nbrs = np.array([self.nbrs(edge) for edge in extant_edges], int)
        extant_nbrs_idx = np.array([self.get_idx(edge) for edge in extant_edges], int)

        # from paper: set of N_{u, v} for all edges (u, v)
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
        nbrhood = lambda edge_idx: self.adj_matrix[mask(edge_idx)][:, mask(edge_idx)]
        max_num_edges_in_nbrhood = (
            lambda edge_idx: (nbrhood(edge_idx).sum() - mask(edge_idx).sum()) // 2
        )

        # from paper: set of c_{u, v} for all edges (u, v)
        self.nbrhood_edge_counts = np.array(
            [
                max_num_edges_in_nbrhood(edge_idx)
                for edge_idx in np.arange(max_num_edges)
            ],
            int,
        )

        # important structs are:
        # self.common_neighbors
        # self.nbrhood_edge_counts
        # # and fun is
        # self.nbrs

    @staticmethod
    def n_choose_2(n):
        return n * (n - 1) // 2

    def reducible_copy(self):
        return ReducibleUndDepGraph(self)


class ReducibleUndDepGraph(UndirectedDependenceGraph):
    def __init__(self, udg):
        self.unreduced = udg.unreduced if hasattr(udg, "unreduced") else udg
        self.adj_matrix = udg.adj_matrix.copy()
        self.num_vertices = udg.num_vertices
        self.num_edges = udg.num_edges

        self.the_cover = None
        self.verbose = udg.verbose

        # from auxilliary structure if needed
        if not hasattr(udg, "get_idx"):
            udg.make_aux()
        self.get_idx = udg.get_idx

        # need to also update these all when self.cover_edges() is called? already done in rule_1
        self.common_neighbors = udg.common_neighbors.copy()
        self.nbrhood_edge_counts = udg.nbrhood_edge_counts.copy()

        # update when cover_edges() is called, actually maybe just extant_edges?
        self.extant_edges_idx = udg.extant_edges_idx.copy()
        self.nbrs = udg.nbrs
        self.get_edge = udg.get_edge

        if hasattr(udg, "reduced_away"):  # then rule_3 wasn't applied
            self.reduced_away = udg.reduced_away

    def reset(self):
        self.__init__(self.unreduced)

    def reduzieren(self, k_num_cliques):
        # reduce by first applying rule 1 and then repeatedly applying
        # rule 2 or rule 3 (both of which followed by rule 1 again)
        # until they don't apply
        if self.verbose:
            print("\t\treducing:")
        self.k_num_cliques = k_num_cliques
        self.reducing = True
        while self.reducing:
            self.reducing = False
            self.rule_1()
            self.rule_2()
            if self.k_num_cliques < 0:
                return
            if self.reducing:
                continue
            self.rule_3()

    def rule_1(self):
        # rule_1: Remove isolated vertices and vertices that are only
        # adjacent to covered edges

        isolated_verts = np.where(self.adj_matrix.sum(0) + self.adj_matrix.sum(1) == 2)[
            0
        ]
        if len(isolated_verts) > 0:  # then Rule 1 is applied
            if self.verbose:
                print("\t\t\tapplying Rule 1...")

            # update auxilliary attributes; LEMMA 2
            self.adj_matrix[isolated_verts, isolated_verts] = 0
            self.num_vertices -= len(isolated_verts)

            # decrease nbrhood edge counts
            for vert in isolated_verts:
                open_nbrhood = self.unreduced.adj_matrix[vert].copy()
                open_nbrhood[vert] = 0
                idx_nbrhoods_to_update = np.where(self.common_neighbors[:, vert] == 1)[
                    0
                ]
                tiled = np.tile(
                    open_nbrhood, (len(idx_nbrhoods_to_update), 1)
                )  # instead of another loop
                to_subtract = np.logical_and(
                    tiled, self.common_neighbors[idx_nbrhoods_to_update]
                ).sum(1)
                self.nbrhood_edge_counts[idx_nbrhoods_to_update] -= to_subtract
                # my own addition:
                # self.nbrhood[:, vert] = 0

            # remove isolated_verts from common neighborhoods
            self.common_neighbors[:, isolated_verts] = 0

        # max_num_edges = self.n_choose_2(self.unreduced.max_num_verts)
        # mask = lambda edge_idx: np.array(self.common_neighbors[edge_idx], dtype=bool)

        # # make subgraph-adjacency matrix, and then subtract diag and
        # # divide by two to get num edges in subgraph---same as sum() of
        # # triu(subgraph-adjacency matrix) but probably a bit faster
        # nbrhood = lambda edge_idx: self.adj_matrix[mask(edge_idx)][:, mask(edge_idx)]
        # max_num_edges_in_nbrhood = lambda edge_idx: (nbrhood(edge_idx).sum() - mask(edge_idx).sum()) // 2

        # # from paper: set of c_{u, v} for all edges (u, v)
        # self.nbrhood_edge_counts = np.array([max_num_edges_in_nbrhood(edge_idx) for edge_idx in np.arange(max_num_edges)], int)
        # # assert (nbrhood_edge_counts==self.nbrhood_edge_counts).all()
        # # print(nbrhood_edge_counts, self.nbrhood_edge_counts)
        # # need to fix!!!!!!!! update isn't working; so just recomputing for now
        # # # # # # # # but actually update produces correct result though recomputing doesn't?

    def rule_2(self):
        # rule_2: If an uncovered edge {u,v} is contained in exactly
        # one maximal clique C, i.e., the common neighbors of u and v
        # induce a clique, then add C to the solution, mark its edges
        # as covered, and decrease k by one

        score = self.n_choose_2(self.common_neighbors.sum(1)) - self.nbrhood_edge_counts
        # score of n implies edge is in exactly n+1 maximal cliques,
        # so we want edges with score 0

        clique_idxs = np.where(score[self.extant_edges_idx] == 0)[0]

        if clique_idxs.size > 0:
            clique_idx = clique_idxs[-1]
            if self.verbose:
                print("\t\t\tapplying Rule 2...")
            clique = self.common_neighbors[self.extant_edges_idx[clique_idx]].copy()
            self.the_cover = (
                clique.reshape(1, -1)
                if self.the_cover is None
                else np.vstack((self.the_cover, clique))
            )
            self.cover_edges()
            self.k_num_cliques -= 1
            self.reducing = True
        # start the loop over so Rule 1 can 'clean up'
        # self.common_neighbors[clique_idxs[0]] = 0  # zero out row, to update struct? not in paper?

    def rule_3(self):
        # rule_3: Consider a vertex v that has at least one
        # prisoner. If each prisoner is connected to at least one
        # vertex other than v via an uncovered edge (automatically
        # given if instance is reduced w.r.t. rules 1 and 2), and the
        # prisoners dominate the exit,s then delete v. To reconstruct
        # a solution for the unreduced instance, add v to every clique
        # containing a prisoner of v.

        for vert, nbrhood in enumerate(self.adj_matrix):
            if nbrhood[vert] == 0:  # then nbrhood is empty
                continue
            exits = np.zeros(self.unreduced.max_num_verts, bool)
            nbrs = np.flatnonzero(nbrhood)
            for nbr in nbrs:
                if (nbrhood.astype(int) - self.adj_matrix[nbr].astype(int) == -1).any():
                    exits[nbr] = True
            # exits[j] == True iff j is an exit for vert
            # prisoners[j] == True iff j is a prisoner of vert
            prisoners = np.logical_and(~exits, self.adj_matrix[vert])
            prisoners[vert] = False  # a vert isn't its own prisoner

            # check if each exit is adjacent to at least one prisoner
            prisoners_dominate_exits = self.adj_matrix[prisoners][:, exits].sum(0)
            if prisoners_dominate_exits.all():  # apply the rule
                self.reducing = True
                if self.verbose:
                    print("\t\t\tapplying Rule 3...")

                # mark edges covered so rule_1 will delete vertex
                edges = np.array(
                    [
                        [vert, u]
                        for u in np.flatnonzero(self.adj_matrix[vert])
                        if vert != u
                    ]
                )
                edges.sort()  # so they're in triu, so that self.get_idx works
                self.cover_edges(edges)

                # keep track of deleted nodes and their prisoners for reconstructing solution
                if not hasattr(self, "reduced_away"):
                    self.reduced_away = np.zeros_like(self.adj_matrix, bool)
                self.reduced_away[vert] = prisoners
                break

    def choose_nbrhood(self):
        # only compute for existing edges
        common_neighbors = self.common_neighbors[self.extant_edges_idx]
        nbrhood_edge_counts = self.nbrhood_edge_counts[self.extant_edges_idx]
        score = self.n_choose_2(common_neighbors.sum(1)) - nbrhood_edge_counts

        # idx in reduced idx list, not from full edge list
        chosen_edge_idx = score.argmin()

        mask = common_neighbors[chosen_edge_idx].astype(bool)

        chosen_nbrhood = self.adj_matrix.copy()
        chosen_nbrhood[~mask, :] = 0
        chosen_nbrhood[:, ~mask] = 0
        if chosen_nbrhood.sum() <= mask.sum():
            raise ValueError("This nbrhood has no edges")
        return chosen_nbrhood

    def cover_edges(self, edges=None):
        if edges is None:
            # always call after updating the cover; only on single, recently added clique
            if self.the_cover is None:
                return  # self.adj_matrix

            # change edges to 0 if they're covered
            clique = self.the_cover[-1]
            covered = np.where(clique)[0]

            # trick for getting combinations from idx
            comb_idx = np.triu_indices(len(covered), 1)

            # actual pairwise combinations; ie all edges (v_i, v_j) covered by the clique
            covered_edges = np.empty((len(comb_idx[0]), 2), int)
            covered_edges[:, 0] = covered[comb_idx[0]]
            covered_edges[:, 1] = covered[comb_idx[1]]

        else:
            covered_edges = edges

        # cover (remove from reduced_graph) edges
        self.rm_edges(covered_edges)
        # update extant_edges_idx
        rmed_edges_idx = [self.get_idx(edge) for edge in covered_edges]
        extant_rmed_edges_idx = [
            edge for edge in rmed_edges_idx if edge in self.extant_edges_idx
        ]
        idx_idx = np.array(
            [np.where(self.extant_edges_idx == idx) for idx in extant_rmed_edges_idx],
            int,
        ).flatten()

        self.extant_edges_idx = np.delete(self.extant_edges_idx, idx_idx)
        # now here do all the updates to nbrs?----actually probably don't want this? see 2clique house example

        # update self.common_neighbors
        # self.common_neighbors[rmed_edges_idx] = 0   # zero out rows covered edges; maybe not necessary, since common_neighbors is (probably?) only called with extant_edges_idx?

        if self.verbose:
            print("\t\t\t{} uncovered edges remaining".format(self.num_edges))

        # if edges is None:
        #     cover_orig = self.the_cover
        #     while self.the_cover.shape[0] > 1:
        #         self.the_cover = self.the_cover[:-1]
        #         self.cover_edges()
        #     self.the_cover = cover_orig

    def reconstruct_cover(self, the_cover):
        if not hasattr(self, "reduced_away"):  # then rule_3 wasn't applied
            return the_cover

        # add the reduced away vert to all covering cliques containing at least one of its prisoners
        to_expand = np.flatnonzero(self.reduced_away.sum(1))
        for vert in to_expand:
            prisoners = self.reduced_away[vert]
            tiled_prisoners = np.tile(
                prisoners, (len(the_cover), 1)
            )  # instead of another loop
            cliques_to_update_mask = (
                np.logical_and(tiled_prisoners, the_cover).sum(1).astype(bool)
            )
            the_cover[cliques_to_update_mask, vert] = 1

        return the_cover


# class minMCM(object):
# TODO: implement as a bigraph with biadjacency matrix with rows M and cols L


# class MCM(object):
# TODO: implement as large DAG adj matrix over L and M, or smaller bigraph for L-M connections and DAG adj matirx for L->L connections
