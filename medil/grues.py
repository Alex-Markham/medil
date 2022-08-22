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

    def grues(self, init="empty", max_repeats=10):
        self.init_uec(init)
        self.get_max_cpdag()
        self.get_score = GaussObsL0Pen(self.samples)
        self.score = self.get_score.full(self.cpdag)
        self.reduce_max_cpdag()
        self.repeated = 0
        while self.repeated < max_repeats:
            print(str(max_repeats - self.repeated) + " repeats left")
            self.old_cpdag = np.copy(self.cpdag)
            self.old_dag = np.copy(self.dag_reduction)
            self.old_cc = np.copy(self.chain_comps)

            move = np.random.choice((self.merge, self.split, self.fiber))
            try:
                move()
            except ValueError:
                self.repeated += 1
                continue

            if self.debug:
                # each vertex is a chain component
                assert len(self.chain_comps) == len(self.dag_reduction)
                assert (self.chain_comps.sum(0) == 1).all()

                # every child has a unique parent set with at least two parents
                num_pars = self.dag_reduction.sum(0)
                children = self.dag_reduction[:, num_pars.astype(bool)]
                assert (np.unique(children, axis=1) == children).all()
                assert (num_pars != 1).all()

                # indeed a DAG and trasitively closed
                num_nodes = len(self.dag_reduction)
                graph = np.copy(self.dag_reduction, int)
                for n in range(2, num_nodes):
                    graph += np.linalg.matrix_power(self.dag_reduction, n)
                assert np.diag(graph).sum() == 0
                assert (graph.astype(bool) == self.dag_reduction).all()

                # check interesection number
                old_intersection_num = np.logical_not(obj.old_dag.sum(0)).sum()
                new_intresection_num = np.logical_not(obj.dag_reduction.sum(0)).sum()
                if move is self.merge:
                    assert old_intersection_num - 1 == new_intresection_num
                elif move is self.split:
                    assert old_intersection_num + 1 == new_intresection_num
                else:
                    assert old_intersection_num == new_intresection_num

                self.expand()
                new_score = self.get_score.full(self.cpdag)
                if new_score < self.score:
                    self.score = new_score
                    self.repeated = 0
                else:
                    self.cpdag = self.old_cpdag
                    self.dag_reduction = self.old_dag
                    self.chain_comps = self.old_cc
                    self.repeated += 1

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
        src_1, src_2 = self.pick_source_nodes("merge")
        self.perform_merge(src_1, src_2)

    def perform_merge(self, src_1, src_2, recurse=True):
        self.chain_comps[src_1] += self.chain_comps[src_2]
        self.dag_reduction[src_1] += self.dag_reduction[src_2]

        self.chain_comps = np.delete(self.chain_comps, src_2, 0)
        self.dag_reduction = np.delete(self.dag_reduction, src_2, 0)
        self.dag_reduction = np.delete(self.dag_reduction, src_2, 1)

        if recurse:
            src_1 -= 1 if src_1 > src_2 else 0
            parentless = np.flatnonzero(self.dag_reduction.sum(0) == 1)
            for child in parentless:
                self.perform_merge(src_1, child, False)

    def split(self):
        v, w, source = self.consider_split()
        self.perform_split(v, w, source)

    def consider_split(self):
        # uniformly pick a source chain component to split
        source = self.pick_source_nodes("split")

        # uniformly pick edge v--w in the chain component to split on
        chain_comp_mask = self.chain_comps[source]
        v, w = np.random.choice(np.flatnonzero(chain_comp_mask), 2, replace=False)

        return v, w, source

    def perform_split(self, v, w, source, recurse=True, fiber=False):
        # add node to dag reduction and corresponding cc to chain comps
        v_cc_mask = np.zeros((1, self.num_feats), bool)
        v_cc_mask[0, v] = 1
        self.chain_comps[source, v] = 0
        self.chain_comps = np.vstack((self.chain_comps, v_cc_mask))
        col = np.zeros((len(self.dag_reduction), 1), bool)
        self.dag_reduction = np.hstack((self.dag_reduction, col))
        # add edges from v_node to children of source node
        if not fiber:
            self.dag_reduction = np.vstack(
                (self.dag_reduction, self.dag_reduction[source])
            )

        if recurse and self.chain_comps[source].sum() != 1:
            self.perform_split(w, None, source, False)
            self.dag_reduction[-2:, source] = 1

    def fiber(self):
        within, src_1, src_2, t, v = self.consider_fiber()
        self.perform_fiber(within, src_1, src_2, t, v)

    def consider_fiber(self):
        within = np.random.choice((True, False))

        src_1, src_2 = self.pick_source_nodes("fiber")

        ch_src_1, ch_src_2 = self.dag_reduction[[src_1, src_2], :]
        poss_t_mask = np.logical_and(ch_src_1, ~ch_src_2)
        poss_t_mask[src_1] = self.chain_comps[src_1].sum() > 1
        t = np.random.choice(np.flatnonzero(poss_t_mask))

        v = np.random.choice(np.flatnonzero(self.chain_comps[t]))

        return within, src_1, src_2, t, v

    def perform_fiber(self, within, src_1, src_2, t, v):
        ch_src_1, ch_src_2 = self.dag_reduction[[src_1, src_2], :]
        ch_intrx_mask = np.logical_and(ch_src_1, ch_src_2)
        num_pars = self.dag_reduction[:, ch_intrx_mask].sum(0)
        exclusive_ch = np.flatnonzero(ch_intrx_mask)[np.flatnonzero(num_pars == 2)]
        if t == src_1:
            if len(exclusive_ch) == 1:
                if within:
                    self.chain_comps[exclusive_ch, v] = True
                else:
                    self.chain_comps[src_2, v] = True
                self.chain_comps[t, v] = False  # perform_split does this elsewhere
            else:
                self.perform_split(v, None, t, False, True)
                self.dag_reduction[src_2, -1] = True
                ch_intrx_mask = np.append(ch_intrx_mask, False)
                self.dag_reduction = np.vstack((self.dag_reduction, ch_intrx_mask))
                if within:
                    self.dag_reduction[src_1, -1] = True
        elif self.chain_comps[t].sum() > 1:
            self.perform_split(v, None, t, False, True)
            v_ch_mask = np.zeros((1, len(self.chain_comps)), bool)
            self.dag_reduction = np.vstack((self.dag_reduction, v_ch_mask))
            self.dag_reduction[t, -1] = True
            t_pars = np.flatnonzero(self.dag_reduction[:, t])
            if not within:
                t_pars = t_pars[t_pars != src_1]
            self.dag_reduction[t_pars, -1] = True
            if len(exclusive_ch) == 1:
                self.dag_reduction[exclusive_ch, -1] = True
        else:
            self.dag_reduction[src_2, t] = True
            if within:
                self.dag_reduction[src_1, t] = False
            elif len(exclusive_ch) == 1:
                self.dag_reduction[exclusive_ch[0], t] = True

    def pick_source_nodes(self, move):
        # sources have no parents; sinks have parents and no children
        non_srcs_mask = self.dag_reduction.sum(0).astype(bool)
        sources = np.flatnonzero(np.logical_not(non_srcs_mask))
        childless_mask = np.logical_not(self.dag_reduction.sum(1).astype(bool))
        sinks = np.flatnonzero(np.logical_and(non_srcs_mask, childless_mask))
        if move == "merge":
            num_sources = self.dag_reduction[np.ix_(sources, sinks)].sum(0)
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
        return np.vectorize(lambda n: math.comb(n, 2), otypes=[int])(
            np.array(vec, ndmin=1)
        )

    def reduce_max_cpdag(self):
        cpdag = np.copy(self.cpdag)
        undir = np.logical_and(cpdag, cpdag.T)
        chain_comps = np.eye(self.num_feats).astype(bool)
        while undir.any() and len(undir) > 1:
            v, w = np.unravel_index(undir.argmax(), undir.shape)
            cpdag = np.delete(cpdag, v, 0)
            cpdag = np.delete(cpdag, v, 1)
            undir = np.delete(undir, v, 0)
            undir = np.delete(undir, v, 1)
            chain_comps[w] += chain_comps[v]
            chain_comps = np.delete(chain_comps, v, 0)

        self.dag_reduction = cpdag
        self.chain_comps = chain_comps

    def expand(self):
        self.cpdag = np.zeros_like(self.cpdag)
        for (pa, ch) in np.argwhere(self.dag_reduction):
            pa_mask, ch_mask = self.chain_comps([pa, cha])
            self.cpdag[pa_mask, ch_mask] = True

    # for scoring, just save the old reduction and chain comps and expansion, make the move, expand self.cpdag, and compare scores of old and new expansion
