"""Implement the Groebner basis-based UEC search algorithm (GrUES)."""
import numpy as np
import math


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
        self.reduce_max_cpdag()
        stop_condition = False  # note: need to figure out what this should be
        while max_iter or not stop_condition:
            max_iter -= 1
            pass

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
        i, k = self.pick_cliques()

        # perform merge and update dag reduction and chain components
        self.chain_comps[k] += chain_components[i]
        self.chain_comps = np.delete(chain_components, i, 0)
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
        v, w = np.random.choice(np.flatnonzero(chosen_clique), 2)

        # perform split and update dag reduction and chain components
        v_clique = w_clique = chosen_clique
        v_clique[w] = w_clique[v] = 0
        self.chain_comps[chosen_clique_idx] = v_clique
        # dag reduction still correct, since v_clique has same sinks
        # as chosen clique; now update for w_clique:
        self.chain_comps = np.vstack((self.chain_comps, w_clique))
        new_edge = np.array([len(self.chain_comps), self.dag_reduction[, 1]])
        self.dag_reduction = np.vstack((self.dag_reduction, new_edge))

    def within_fiber(self):
        pass

    def without_fiber(self):
        pass

    def pick_cliques(self):
        r"""Finds i, k such that there is v-structure i -> j <- k."""

        # pick j from uniform distribution over v-structures i -> j <- k
        n_choose_2 = np.vectorize(lambda n: math.comb(n, 2))
        counts = n_choose_2(self.dag_reduction.sum(0))
        p = counts / counts.sum()
        j = np.random.choice(np.arange(len(p)), p)

        # pick i and k uniformly
        pa_j = np.argwhere(self.dag_reduction[:, cj])
        i, k = np.random.choice(pa_j, 2)

        return i, k

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
            chain_comps[w] += chain_components[v]
            chain_comps = np.delete(chain_components, v, 0)

        self.dag_reduction = np.argwhere(cpdag)
        self.chain_comps = chain_comps
