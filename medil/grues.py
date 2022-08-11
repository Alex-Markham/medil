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

    def __init__(self, samples):
        self.samples = np.array(samples, dtype=float)
        self.num_samps, self.num_feats = self.samples.shape
        self.debug = False

    def grues(self, init="empty", max_iter=100):
        self.init_uec(init)
        self.get_max_cpdag()
        self.get_score = GaussObsL0Pen(self.samples)
        self.score = self.get_score.full(self.cpdag)
        self.reduce_max_cpdag()
        stop_condition = False  # note: need to figure out what this should be
        while max_iter or not stop_condition:
            max_iter -= 1
            move = np.random.choice(
                (self.merge, self.split, self.within_fiber, self.out_of_fiber)
            )
            move()

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
        r"""Merges cliques corresponding to i, k connected by
        v-structure i -> j <- k in dag reduction."""

        # uniformaly pick a pair of cliques to consider for merge
        i, k, j = self.pick_source_ccs(merge)

        # score and perform merge
        other_pa = self.get_other_parents(j, i, k)
        score_update = self.score_of_merge(i, k, j, other_pa)
        if score_update >= 0:
            self.score += score_update
            self.perform_merge(i, k, j, len(other_pa))
        else:
            return

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

    def perform_merge(self, i, k, j, other_pa):
        r"""Merges i into k, updating dag reduction and chain components."""
        self.chain_comps[k] += self.chain_comps[i]
        self.chain_comps = np.delete(self.chain_comps, i, 0)
        to_delete = np.flatnonzero(self.dag_reduction[:, 0] == i)
        self.dag_reduction = np.delete(self.dag_reduction, to_delete, 0)
        ## test the two following
        if not other_pa:  # reduce newly formed cc
            self.chain_comps[k] += self.chain_comps[j]
            self.chain_comps = np.delete(self.chain_comps, j, 0)
            to_delete = np.flatnonzero(self.dag_reduction[:, 1] == j)
            self.dag_reduction = np.delete(self.dag_reduction, to_delete, 0)
        # add children of k to i
        self.dag_reduction[self.dag_reduction == k] = i

    def split(self):
        r"""Splits a clique containing edge v--w, making ne(v) \cap ne(w) into v-structures."""
        considered = self.consider_split()
        score_update = self.score_of_split(considered)
        if score_update >= 0:
            self.score += score_update
            self.perform_split(considered)
        else:
            return

    def consider_split(self):
        # uniformly pick a source chain component to split
        chosen_cc_idx = self.pick_source_cc("split")

        # uniformly pick edge v--w in the clique to split on
        chosen_cc = self.chain_comps[chosen_cc_idx]
        v, w = np.random.choice(np.flatnonzero(chosen_cc), 2, replace=False)

        return v, w, chosen_cc_idx

    def score_of_split(self, considered):
        v, w, chosen_cc_idx = considered
        vw_cc = np.flat_nonzero(self.chain_comps[chosen_cc_idx])
        old = (self.get_score.local(vw, vw_cc[vw_cc != vw]) for vw in vw_cc).sum()
        v_cc = np.delete(vw_cc, w)
        w_cc = np.delete(vw_cc, v)
        new = (self.get_score.local(v, v_cc[v_cc != v]) for v in v_cc).sum()
        new += (self.get_score.local(w, w_cc[w_cc != w]) for w in w_cc).sum()
        return new - old

    def perform_split(self, v, w, chosen_cc_idx):
        # perform split and update dag reduction and chain components
        w_clique = np.copy(self.chain_comps[chosen_cc_idx])
        self.chain_comps[chosen_cc_idx, w] = 0
        w_clique[v] = 0
        # dag reduction for v_clique still correct, since it has same
        # sinks as chosen clique; now update for w_clique:
        self.chain_comps = np.vstack((self.chain_comps, w_clique))
        ch_v_idx = np.flatnonzero(dag_reduction[:, 0] == v)
        ch_w = np.copy(self.dag_reduction[ch_v_idx])
        w_idx = len(self.chain_comps) - 1
        ch_w[:, 0] = w_idx
        self.dag_reduction = np.vstack((self.dag_reduction, ch_w))

    def within_fiber(self):
        i_cc_idx, j_cc_idx, child_cc_idx = self.pick_source_ccs(fiber)
        i, j, t = self.pick_nodes(i_cc_idx, j_cc_idx)

        # score of move
        old = (self.get_score.local(i, i_cc[i_cc != i]) for i in i_cc).sum()
        old += (self.get_score.local(j, j_cc[j_cc != j]) for j in j_cc).sum()
        it_cc = np.append(i_cc, t)
        jt_cc = j_cc[j_cc != t]
        new = (self.get_score.local(it, it_cc[it_cc != it]) for it in it_cc).sum()
        new += (self.get_score.local(jt, jt_cc[jt_cc != jt]) for jt in jt_cc).sum()
        children = self.dag_reduction[self.dag_reduction[:, 0] == i][:, 1]
        children = children[children != j]
        if children.any():
            childs = np.flatnonzero(children)
            other_pars = self.dag_reduction[self.dag_reduction[:, 1] == j][:, 0]
            other_pars = other_pars[other_pars != i]
            other_pars = np.flatnonzero(self.chain_comps[other_pars].sum(0))
            pars = np.append(i_cc, other_pars)
            old += (self.get_score.local(child, pars) for child in childs).sum()
            pars = np.append(other_pars, it_cc)
            new += (self.get_score.local(child, pars) for child in childs).sum()
        score_update = new - old

        if score_update >= 0:  # then perform move
            self.score += score_update
            # transfer t to clique_i
            self.chain_comps[k, t] = 0
            self.chain_comps[i, t] = 1
        else:
            return

    def out_of_fiber(self):
        i_cc_idx, j_cc_idx, child_cc_idx = self.pick_source_ccs(fiber)
        i, j, t = self.pick_nodes(i_cc_idx, j_cc_idx)

        # score of move
        old = self.get_score.local(t, j_cc[j_cc != t])
        old += (self.get_score.local(i, i_cc[i_cc != i]) for i in i_cc).sum()
        it_cc = np.append(i_cc, t)
        new = (self.get_score.local(it, it_cc[it_cc != it]) for it in it_cc).sum()

        score_update = new - old

        if score_update >= 0:  # then perform move
            self.score += score_update
            # add t to clique_i
            self.chain_comps[i, t] = 1
        else:
            return

    def pick_source_ccs(self, move):
        all_sinks = self.dag_reduction[:, 1]  # all sinks that are not also sources
        if move == "merge":
            sinks = np.unique(all_sinks)
            num_pars = np.fromiter(((all_sinks == sink).sum() for sink in sinks), int)
            counts = self.n_choose_2(num_pars)
            p = counts / counts.sum()
            j = np.random.choice(sinks, replace=False, p=p)
            # pick i and k uniformly:
            pa_j = self.dag_reduction[all_sinks == j, 0]
            i, k = np.random.choice(pa_j, 2, replace=False)
            chosen_ccs_idx = i, k, j
        else:  # then move == "split" or "fiber"
            two_plus_cc_idx = np.flatnonzero(self.chain_comps.sum(1) > 1)
            tp_sources_mask = np.logical_not(np.in1d(two_plus_cc_idx, all_sinks))
            splittable_idx = two_plus_cc_idx[tp_sources_mask]
            chosen_ccs_idx = np.random.choice(splittable_idx)
            if move == "fiber":
                j_cc_idx = chosen_ccs_idx
                all_cc_idx = np.arange(len(self.chain_comps) - 1)
                source_ccs_idx = np.logical_not(np.in1d(all_cc_idx, all_sinks))
                source_ccs_idx = source_ccs_idx[source_ccs_idx != j_cc_idx]
                i_cc_idx = np.random.choice(source_ccs_idx)

                ch_i = self.get_other_children(i_cc_idx, None)
                ch_j = self.get_other_children(j_cc_idx, None)
                child_cc_idx = ch_i[np.in1d(ch_i, ch_j)]

                chosen_ccs_idx = i_cc_idx, j_cc_idx, child_cc_idx
        return chosen_ccs_idx

    def pick_nodes(self, i_cc_idx, j_cc_idx):
        # uniformly pick elements i, j from the ccs
        i_cc, j_cc = np.argwhere(self.chain_comps[i_cc_idx, j_cc_idx]).T
        i, j = np.random.choice(i_cc), np.random.choice(j_cc)

        # uniformly pick element t != j from
        # a uniformly chosen cc in (clique of j \ clique of i)
        ch_j_mask = np.flatnonzero(self.dag_reduction[:, 0] == j)
        ch_j_idcs = self.dag_reduction[ch_j_mask, 1]
        ch_j_idcs = ch_j_idcs[ch_j_idcs != j]
        ch_i_mask = np.flatnonzero(self.dag_reduction[:, 0] == i)
        ch_i_idcs = self.dag_reduction[ch_i_mask, 1]
        cc_idcs = np.logical_not(np.in1d(ch_i_idcs, ch_j_idcs))
        chosen_cc_idx = np.random.choice(np.append(cc_idcs, j_cc))
        t = np.random.choice(np.flat_nonzero(self.chain_comps[chosen_cc_idx]))

        return i, j, t

    @staticmethod
    def n_choose_2(vec):
        return np.vectorize(lambda n: math.comb(n, 2))(vec)

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

        self.dag_reduction = np.argwhere(cpdag)
        self.chain_comps = chain_comps

    def get_other_children(self, parent, child, reduced=True):
        children_mask = self.dag_reduction[:, 0] == parent
        children = self.dag_reduction[children_mask, 1]
        children = children[children != child]
        if reduced:
            return children
        else:
            return np.where(self.chain_comps[children])[1]

    def get_other_parents(self, child, pa_1, pa_2=None, reduced=True):
        parents_mask = self.dag_reduction[:, 1] == child
        parents = self.dag_reduction[parent_mask, 0]
        parents = parents[parents != pa_1]
        if pa_2 is not None:
            parents = parents[parents != pa_2]
        if reduced:
            return parents
        else:
            return np.where(self.chain_comps[parents])[1]
