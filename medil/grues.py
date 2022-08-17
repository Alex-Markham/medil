"""Implement the Groebner basis-based UEC search algorithm (GrUES)."""
import numpy as np
import math
from .gauss_obs_l0_pen import GaussObsL0Pen


class InputData(object):
    r"""Feed data into GrUES to learn its UEC.

    Attributes
    ----------
    samples : array_like
        Passes directly to numpy.array to get a :math:`N \times M`
        matrix with :math:`N` samples of :math:`M` random variables.

    Methods
    -------
    grues()
        Return optimal undirected graph representing an unconditional
        equivalence class of DAGs.

    """

    def __init__(self, samples, debug=False):
        self.samples = np.array(samples, dtype=float)
        self.num_samps, self.num_feats = self.samples.shape
        self.debug = debug

    def grues(self, init="empty", max_repeats=5):
        self.init_uec(init)
        self.get_max_cpdag()
        self.get_score = GaussObsL0Pen(self.samples)
        self.score = self.get_score.full(self.cpdag)
        self.reduce_max_cpdag()
        self.repeated = 0
        while self.repeated < max_repeats:
            move = np.random.choice((self.merge, self.split, self.fiber))
            move()
            if self.debug:
                # each vertex is a chain component
                assert len(self.chain_comps) == len(self.dag_reduction)
                assert (self.chain_comps.sum(0) == 1).all()

                # every child has a unique parent set with at least two parents
                num_pars = self.dag_reduction.sum(0)
                children = self.dag_reduction(:, num_pars.astype(bool))
                assert (np.unique(children, axis=1) == children).all()
                assert (num_pars != 1).all()

                # indeed a DAG and trasitively closed
                num_nodes = len(self.dag_reduction)
                graph = np.copy(self.dag_reduction, int)
                for n in range(2, num_nodes):
                    graph += np.linalg.matrix_power(self.dag_reduction, n)
                assert np.diag(graph).sum() == 0
                assert (graph.astype(bool) == self.dag_reduction).all()

    def init_uec(self, init):
        if type(init) is str:
            if init == "empty":
                self.uec = np.zeros((self.num_feats, self.num_feats), bool)
            elif init == "complete":
                self.uec = np.ones((self.num_feats, self.num_feats), bool)
            elif init == "dcov_fast":
                pass  # also add gauss
        else:
            uec = np.array(init, bool)
            is_uec = True  # add actual check of uec-ness
            if not is_uec:
                raise ValueError("Invalid init parameter.")
            self.uec = uec

    def get_max_cpdag(self):
        r"""Return maximal CPDAG in the UEC."""

        # find all induced 2-paths i--j--k, by implicitly looping
        # through missing edges
        i, k = np.triu(np.logical_not(self.uec), 1).nonzero()
        i_or_k_idx, j = np.logical_and(self.uec[i], self.uec[k]).nonzero()

        # remove entries (j, i) and (j, k), thus directing essential
        # edges and removing edges violating implied conditional
        # independence relations
        cpdag = np.copy(self.uec)
        cpdag[j, i[i_or_k_idx]] = cpdag[j, k[i_or_k_idx]] = 0
        self.cpdag = cpdag

    def merge(self):
        # uniformaly pick a pair of cliques to consider for merge
        src_1, src_2 = self.pick_source_nodes("merge")

        # score and perform merge
        score_update = self.score_of_merge(src_1, src_2)
        if score_update >= 0:
            self.score += score_update
            self.perform_merge(src_1, src_2)
            self.repeated = 0
        else:
            self.repeated += 1

    def score_of_merge(self, i, k, j, other_pa):
        i_cc, k_cc = np.argwhere(self.chain_comps[i, k]).T
        old = (self.get_score.local(i, i_cc[i_cc != i]) for i in i_cc).sum()
        old += (self.get_score.local(k, k_cc[k_cc != k]) for k in k_cc).sum()
        ik_cc = np.append(i_cc, k_cc)
        if len(other_pa):
            ik_cc = np.append(ik_cc, np.flatnonzero(self.chain_comps(j)))
        new = (self.get_score.local(ik, ik_cc[ik_cc != ik]) for ik in ik_cc).sum()
        for pa in (i, k):
            for ch in self.get_other_children(pa, j):
                ch_cc = np.flatnonzero(self.chain_comps[ch])
                parents = self.get_other_parents(ch, pa, reduced=False)
                old += (self.get_score.local(child, parents) for child in ch_cc).sum()
                parents = np.append(parents, ik_cc)
                new += (self.get_score.local(child, parents) for child in ch_cc).sum()
        return new - old

    def perform_merge(self, src_1, src_2, recurse=True):
        self.chain_comps[src_1] += self.chain_comps[src_2]
        self.dag_reduction[src_1] += self.dag_reduction[src_2]

        self.chain_comps = np.delete(self.chain_comps, src_2, 0)
        self.dag_reduction = np.delete(self.dag_reduction, src_2, 0)

        if recurse:
            parentless = np.flatnonzero(self.dag_reduction.sum(0) == 1)
            if len(parentless == 1):
                self.perform_merge(src_1, parentless[0], False)

    def split(self):
        considered = self.consider_split()
        score_update = self.score_of_split(considered)
        if score_update >= 0:
            self.score += score_update
            self.perform_split(considered)
            self.repeated = 0
        else:
            self.repeated += 1

    def consider_split(self):
        # uniformly pick a source chain component to split
        source = self.pick_source_nodes("split")

        # uniformly pick edge v--w in the chain component to split on
        chain_comp_mask = self.chain_comps[source]
        v, w = np.random.choice(np.flatnonzero(chain_comp_mask), 2, replace=False)

        return v, w, source

    def score_of_split(self, considered):
        v, w, chosen_cc_idx = considered
        vw_cc = np.flat_nonzero(self.chain_comps[chosen_cc_idx])
        old = (self.get_score.local(vw, vw_cc[vw_cc != vw]) for vw in vw_cc).sum()
        v_cc = np.delete(vw_cc, w)
        w_cc = np.delete(vw_cc, v)
        new = (self.get_score.local(v, v_cc[v_cc != v]) for v in v_cc).sum()
        new += (self.get_score.local(w, w_cc[w_cc != w]) for w in w_cc).sum()
        return new - old

    def perform_split(self, v, w, source, recurse=True):
        # add node to dag reduction and corresponding cc to chain comps
        v_cc_mask = np.zeros(self.num_feats, bool)
        v_cc_mask[v] = 1
        self.chain_comps[source, v] = 0
        self.chain_comps = np.vstack((self.chain_comps, v_cc_mask))
        col = np.zeros((len(self.dag_reduction), 1), bool)
        self.dag_reduction = np.hstack((self.dag_reduction, col))
        # add edges from v_node to children of source node
        self.dag_reduction = np.vstack((self.dag_reduction, self.dag_reduction[source]))

        if self.chain_comps[source].sum() != 1 and recurse:
            self.perform_split(w, None, source, False)
            self.dag_reduction[-2:, source] = 1

    def fiber(self):
        i_cc_idx, j_cc_idx, child_cc_idx = self.pick_source_ccs(fiber)
        i, j, t, t_cc_idx = self.pick_nodes(i_cc_idx, j_cc_idx)

        if score_update >= 0:  # then perform move
            self.score += score_update
            # transfer t to clique_i
            self.chain_comps[k, t] = 0
            self.chain_comps[i, t] = 1
             self.repeated = 0
        else:
            self.repeated += 1

    def pick_source_nodes(self, move):
        # sources have no parents; sinks have parents and no children
        non_srcs_mask = self.dag_reduction.sum(0).astype(bool)
        sources = np.flatnonzero(np.logical_not(non_srcs_mask))
        childless_mask = np.logical_not(self.dag_reduction.sum(1).astype(bool))
        sinks = np.flatnonzero(np.logical_and(non_srcs_mask, childless_mask))
        if move == "merge":
            num_sources = self.dag_reduction[sources, sinks].sum(0)
            counts = self.n_choose_2(num_sources)
            p = np.array(counts / counts.sum())
            sink = np.random.choice(sinks, p=p)
            srcs_of_sink = sources[self.dag_reduction[sources, sink]]
            src_1, src_2 = np.random.choice(srcs_of_sink, 2, replace=False)
            chosen_nodes = src_1, src_2
        else:  # then move == "split" or "fiber"
            non_singleton_nodes = np.flatnonzero(self.chain_comps.sum(1) > 1)
            ns_sources = sources[np.in1d(sources, non_singleton_nodes)]
            chosen_nodes = np.random.choice(ns_sources)
            if move == "fiber":
                # srcs_mask has entry i,j = 1 if and only if
                # i is nonsingleton or
                # i has children other than j's children
                other_children_mask = self.dag_reduction @ ~self.dag_reduction.T
                other_children_mask[ns_sources] = 1
                srcs_mask = other_children_mask[np.ix_(sources, sources)]
                np.fill_diagonal(srcs_mask, 0)
                chosen_idx = np.random.choice(range(srcs_mask.sum()))
                chosen_nodes = sources[np.argwhere(srcs_mask)[chosen_idx]]
        return chosen_nodes

    @staticmethod
    def n_choose_2(vec):
        return np.vectorize(lambda n: math.comb(n, 2))(np.array(vec, ndmin=1))

    def reduce_max_cpdag(self):
        cpdag = np.copy(self.cpdag)
        undir = np.logical_and(cpdag, cpdag.T)
        chain_comps = np.eye(self.num_feats).astype(bool)
        while undir.any():
            v, w = np.unravel_index(undir.argmax(), undir.shape)
            cpdag = np.delete(cpdag, v, 0)
            cpdag = np.delete(cpdag, v, 1)
            undir = np.delete(undir, v, 0)
            undir = np.delete(undir, v, 1)
            chain_comps[w] += chain_comps[v]
            chain_comps = np.delete(chain_comps, v, 0)

        self.dag_reduction = cpdag
        self.chain_comps = chain_comps
