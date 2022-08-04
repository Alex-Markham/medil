"""Implement the Groebner basis-based UEC search algorithm (GrUES)."""
import numpy as np
import math
from .gauss_obs_l0_pen import GaussObsL0Pen


class InputData(object):
    r"""Feed data into GrUES to learn its UEC.

    Attributes
    ----------
    samples : array_like
        Passes directly to numpy.array to get a :math:`M \times N`
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

    def grues(self, init="empty", max_iter=100):
        self.init_uec(init)
        self.get_max_cpdag()
        self.score_obj = GaussObsL0Pen(self.samples)
        self.score = self.score_obj.full_score(self.cpdag)
        self.reduce_max_cpdag()
        stop_condition = False  # note: need to figure out what this should be
        while max_iter or not stop_condition:
            max_iter -= 1
            move = np.random.choice(
                (self.merge, self.split, self.within_fiber, self.out_of_fiber)
            )
            move()

    def init_uec(self, init):
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
        i, k = np.logical_not(self.uec).nonzero()
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

        # uniformaly pick a pair of cliques to merge
        i, k, j = self.pick_cliques()

        ## score possible move here, then proceed if score improves or abort otherwise
        children, pa_i, pa_k = np.argwhere(self.chain_comps[j, i, k]).T
        old = (self.score_obj.local_score(child, pa_i) for child in children).sum()
        old += (self.score_obj.local_score(child, pa_k) for child in children).sum()
        new = (
            self.score_obj.local_score(child, np.append(pa_i, pa_k))
            for child in children
        ).sum()
        score_change = new - old
        if score_change <= 0:
            return
        else:
            self.score += score_change
            # perform merge and update dag reduction and chain components
            self.chain_comps[k] += self.chain_comps[i]
            self.chain_comps = np.delete(self.chain_comps, i, 0)
            to_delete = np.where(self.dag_reduction[:, 0] == i)[0]
            self.dag_reduction = np.delete(self.dag_reduction, to_delete, 0)

    def split(self):
        r"""Splits a clique containing edge v--w, making ne(v) \cap ne(w) into v-structures."""

        # uniformly pick a clique to split
        two_plus_cliques_idx = np.flatnonzero(self.chain_comps.sum(1) > 1)
        source_cliques_idx = self.dag_reduction[:, 0]
        splittable_mask = np.in1d(source_cliques_idx, two_plus_cliques_idx)
        splittable_idx = source_cliques_idx[splittable_mask]
        chosen_clique_idx = np.random.choice(splittable_idx)
        chosen_clique = self.chain_comps[chosen_clique_idx]

        # uniformly pick edge v--w in the clique to split on
        v, w = np.random.choice(np.flatnonzero(chosen_clique), 2, replace=False)

        ## score possible move here, then proceed if score improves or abort otherwise
        children_mask = np.flatnonzero(chosen_clique)
        old = (self.score_obj.local_score(child, pa_i) for child in children).sum()
        old += (self.score_obj.local_score(child, pa_k) for child in children).sum()
        new = (
            self.score_obj.local_score(child, np.append(pa_i, pa_k))
            for child in children
        ).sum()
        score_change = new - old
        if score_change <= 0:
            return
        else:
            self.score += score_change
            # perform split and update dag reduction and chain components
            v_clique = w_clique = chosen_clique
            v_clique[w] = w_clique[v] = 0
            self.chain_comps[chosen_clique_idx] = v_clique
            # dag reduction still correct, since v_clique has same sinks
            # as chosen clique; now update for w_clique:
            self.chain_comps = np.vstack((self.chain_comps, w_clique))
            new_edge = np.array(
                [len(self.chain_comps), self.dag_reduction[:, 1]]
            )  # : was missing; could be bug here now
            self.dag_reduction = np.vstack((self.dag_reduction, new_edge))

    def within_fiber(self):
        # uniformly pick a pair of cliques
        i, k, j = self.pick_cliques()
        i, k = np.random.choice((i, k), 2, replace=False)

        # uniformly pick element t of clique_k
        t = np.random.choice(np.flatnonzero(self.chain_comps[k]))

        ## score possible move here, then either abort or proceed

        # transfer t to clique_i
        self.chain_comps[k, t] = 0
        self.chain_comps[i, t] = 1

    def out_of_fiber(self):
        # uniformly pick a pair of cliques
        i, k, j = self.pick_cliques()
        i, k = np.random.choice((i, k), 2, replace=False)

        # uniformly pick element t of clique_k
        t = np.random.choice(np.flatnonzero(self.chain_comps[k]))

        ## score possible move here, then either abort or proceed

        # add t to clique_i
        self.chain_comps[i, t] = 1

    def pick_cliques(self):
        r"""Finds i, k such that there is v-structure i -> j <- k."""

        # pick j from uniform distribution over v-structures i -> j <- k
        n_choose_2 = np.vectorize(lambda n: math.comb(n, 2))
        counts = n_choose_2(self.dag_reduction.sum(0))
        p = counts / counts.sum()
        j = np.random.choice(np.arange(len(p)), p, replace=False)

        # pick i and k uniformly
        pa_j = np.argwhere(self.dag_reduction[:, j])
        i, k = np.random.choice(pa_j, 2, replace=False)

        return i, k, j

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
            chain_comps[w] += self.chain_comps[v]
            chain_comps = np.delete(self.chain_comps, v, 0)

        self.dag_reduction = np.argwhere(cpdag)
        self.chain_comps = chain_comps
