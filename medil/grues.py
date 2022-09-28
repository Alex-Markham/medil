"""Implement the Groebner basis-based UEC search algorithm (GrUES)."""
import numpy as np
import math
import warnings
from .gauss_obs_l0_pen import GaussObsL0Pen
from .independence_testing import dcov
from scipy.stats import chi2, beta
from scipy.special import factorial
from numpy.linalg import lstsq, det, inv


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

    def __init__(self, samples, rng=np.random.default_rng(0)):
        self.samples = np.array(samples, dtype=float)
        self.rng = rng
        self.pmf = None

        self.num_samps, self.num_feats = self.samples.shape
        self.explore = False
        self.debug = False

    def mcmc(self, init="empty", p="uniform", max_moves=10000, prior=None):
        if p == "uniform":
            self.p = {
                "merge": 1 / 6,
                "split": 1 / 6,
                "within": 1 / 3,
                "out_del": 1 / 6,
                "out_add": 1 / 6,
            }
        else:
            self.p = p
        self.init_uec(init)
        self.get_max_cpdag()
        self.reduce_max_cpdag()
        self.moves = 0
        self.visited = np.empty(max_moves, float)
        self.old_rss = self.compute_mle_rss()
        self.visited[0] = self.optimal_bic = self.cpdag.sum() * np.log(
            self.num_samps
        ) + self.num_samps * np.log(self.old_rss)
        self.markov_chain = np.empty((max_moves, self.num_feats, self.num_feats), bool)
        self.markov_chain[0] = self.uec
        while self.moves < max_moves - 1:
            self.old_bic = self.visited[self.moves]
            self.old_cpdag = np.copy(self.cpdag)
            self.old_dag = np.copy(self.dag_reduction)
            self.old_cc = np.copy(self.chain_comps)

            moves_dict = {
                "merge": self.merge,
                "split": self.split,
                "within": self.algebraic,
                "out_del": self.algebraic,
                "out_add": self.algebraic,
            }

            poss_moves, considered, p = self.compute_transition_kernel(moves_dict)
            move = self.rng.choice(poss_moves, p=p)

            q = float(self.q[move] * p[poss_moves == move])

            moves_dict[move](considered[move])

            if self.debug:
                try:
                    self.run_checks(move)
                except AssertionError:
                    import pdb, traceback

                    exc = traceback.format_exc()
                    pdb.set_trace()

            inv_moves = {
                "merge": "split",
                "split": "merge",
                "within": "within",
                "out_del": "out_add",
                "out_add": "out_del",
            }
            inv_move = inv_moves[move]

            poss_moves, _, p = self.compute_transition_kernel(moves_dict)
            q_inv = float(self.q[inv_move] * p[poss_moves == inv_move])

            likelihood_ratio, new_rss, new_bic = self.get_likelihood_ratio()
            if new_bic / self.optimal_bic < 1:
                self.optimal_bic = new_bic
                self.optimal_uec = self.uec

            likelihood_and_transition_ratio = likelihood_ratio * (q_inv / q)
            if type(prior) is tuple:
                prior_ratio = self.scaled_triangle_prior(prior)
                likelihood_and_transition_ratio *= prior_ratio
            elif prior is not None:
                prior_ratio = prior(self)
                likelihood_and_transition_ratio *= prior_ratio
            h = min(1, likelihood_and_transition_ratio)
            if self.explore:
                h = 1
            make_move = self.rng.choice((True, False), p=(h, 1 - h))
            if make_move:
                self.old_rss = new_rss
                self.moves += 1
                self.visited[self.moves] = new_bic
                self.markov_chain[self.moves] = self.uec
                # (2 ** np.flatnonzero(self.cpdag)).sum()]
            else:
                self.cpdag = self.old_cpdag
                self.dag_reduction = self.old_dag
                self.chain_comps = self.old_cc

    def init_uec(self, init):
        if type(init) is str:
            if init == "empty":
                self.uec = np.zeros((self.num_feats, self.num_feats), bool)
            elif init == "complete":
                self.uec = np.ones((self.num_feats, self.num_feats), bool)
                np.fill_diagonal(self.uec, False)
        elif type(init) is tuple:
            init, alpha = init
            if init == "gauss":
                corr = np.corrcoef(self.samples, rowvar=False)
                dist = beta(
                    self.num_samps / 2 - 1, self.num_samps / 2 - 1, loc=-1, scale=2
                )
                crit_val = abs(dist.ppf(alpha / 2))
                self.uec = self.indep_test_U = abs(corr) >= crit_val
                np.fill_diagonal(self.uec, False)
            elif init == "dcov_fast":
                cov, d_bars = dcov(self.samples)
                crit_val = chi2(1).ppf(1 - alpha)
                test_val = self.num_samps * cov / np.outer(d_bars, d_bars)
                self.uec = test_val >= crit_val
                np.fill_diagonal(self.uec, False)
        else:
            self.uec = np.array(init, bool)
        self.optimal_uec = self.uec

    def get_max_cpdag(self):
        r"""Return maximal CPDAG in the UEC."""

        U = np.copy(self.uec)
        compliment_U = ~U
        np.fill_diagonal(compliment_U, False)

        # V_ij == 1 if and only if there's a k adjacent to j but not i
        V = compliment_U @ U

        # W_ij == 1 if and only if there's k such that i--j--k is an induced path
        W = np.logical_and(V, U).T

        # This orients all v-structures and removes edges violating CI relations
        U[W] = False

        self.cpdag = U
        recon_uec = self.get_uec()
        if not (recon_uec == self.uec).all():
            # then self.uec was invalid, so reinitialize
            self.uec = self.optimal_uec = recon_uec
            self.get_max_cpdag()

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

    def merge(self, considered):
        src_1, src_2 = considered
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

    def split(self, considered):
        v, w, source = considered
        self.perform_split(v, w, source)

    def consider_split(self):
        # uniformly pick a source chain component to split
        source = self.pick_source_nodes("split")

        # uniformly pick edge v--w in the chain component to split on
        chain_comp_mask = self.chain_comps[source]
        choices = np.flatnonzero(chain_comp_mask)
        v, w = self.rng.choice(choices, 2, replace=False)

        self.q["split"] /= self.n_choose_2(len(choices))

        return v, w, source

    def perform_split(self, v, w, source, recurse=True, algebraic=False):
        # add node to dag reduction and corresponding cc to chain comps
        v_cc_mask = np.zeros((1, self.num_feats), bool)
        v_cc_mask[0, v] = True
        self.chain_comps[source, v] = False
        self.chain_comps = np.vstack((self.chain_comps, v_cc_mask))
        col = np.zeros((len(self.dag_reduction), 1), bool)
        self.dag_reduction = np.hstack((self.dag_reduction, col))
        # add edges from v_node to children of source node
        if not algebraic:
            self.dag_reduction = np.vstack(
                (self.dag_reduction, self.dag_reduction[source])
            )

        if recurse and self.chain_comps[source].sum() != 1:
            self.perform_split(w, None, source, False)
            self.dag_reduction[-2:, source] = True

    def algebraic(self, considered):
        src_1, src_2, t, v, T_mask = considered
        self.perform_algebraic(src_1, src_2, t, v, T_mask)

    def consider_algebraic(self, fiber):
        if fiber == "out_del":
            src_1, t = self.pick_source_nodes("out_del")
            src_2 = None
        else:
            src_1, src_2 = self.pick_source_nodes(fiber)
            ch_src_1, ch_src_2 = self.dag_reduction[[src_1, src_2], :]
            poss_t_mask = np.logical_and(ch_src_1, ~ch_src_2)
            poss_t_mask[src_1] = self.chain_comps[src_1].sum() > 1
            choices = np.flatnonzero(poss_t_mask)
            t = self.rng.choice(choices)
            self.q[fiber] /= len(choices)
            if len(choices) == 0:
                warnings.warn("inaccurate self.q")
        choices = np.flatnonzero(self.chain_comps[t])
        v = self.rng.choice(choices)
        self.q[fiber] /= len(choices)
        if len(choices) == 0:
            warnings.warn("inaccurate self.q")

        if src_1 == t:
            T_mask = np.zeros(len(self.dag_reduction), bool)
            T_mask[t] = True
        else:
            max_anc_mask = self.dag_reduction.sum(0) == 0
            par_t_mask = self.dag_reduction[:, t]
            T_mask = np.logical_and(max_anc_mask, par_t_mask)

        if fiber == "out_del":
            T_mask[src_1] = False
        elif fiber == "out_add":
            T_mask[src_2] = True
        else:  # fiber == "within"
            T_mask[[src_1, src_2]] = False, True

        return src_1, src_2, t, v, T_mask

    def perform_algebraic(self, src_1, src_2, t, v, T_mask):
        nonsources_mask = self.dag_reduction.sum(0).astype(bool)
        max_anc_dag = np.copy(self.dag_reduction)
        max_anc_dag[nonsources_mask] = False
        sources = np.flatnonzero(np.logical_not(nonsources_mask))
        max_anc_dag[sources, sources] = True

        num_common = T_mask.astype(int) @ max_anc_dag
        has_other_ancs = ~T_mask @ max_anc_dag
        P_mask = np.logical_and(num_common.astype(bool), ~has_other_ancs)
        C_mask = num_common == T_mask.sum()
        exact = np.flatnonzero(np.logical_and(P_mask, C_mask))
        if len(exact) == 1:
            self.chain_comps[[exact[0], t], v] = True, False
        else:
            self.perform_split(v, None, t, recurse=False, algebraic=True)
            C_mask = np.append(C_mask, False)
            self.dag_reduction = np.vstack((self.dag_reduction, C_mask))
            P_mask = np.append(P_mask, False)
            self.dag_reduction[P_mask, -1] = True

        if not self.chain_comps[t].any():
            self.chain_comps = np.delete(self.chain_comps, t, 0)
            self.dag_reduction = np.delete(self.dag_reduction, t, 0)
            self.dag_reduction = np.delete(self.dag_reduction, t, 1)

    def pick_source_nodes(self, move):
        # sources have no parents; sinks have parents and no children
        non_srcs_mask = self.dag_reduction.sum(0).astype(bool)
        sources = np.flatnonzero(np.logical_not(non_srcs_mask))
        childless_mask = np.logical_not(self.dag_reduction.sum(1).astype(bool))
        sinks = np.flatnonzero(np.logical_and(non_srcs_mask, childless_mask))
        if move == "merge":
            singleton_nodes = np.flatnonzero(self.chain_comps.sum(1) == 1)
            sngl_srcs = sources[np.in1d(sources, singleton_nodes)]
            ch_subgraph = self.dag_reduction[sngl_srcs].astype(int)
            empty_ch_subgraph = (~self.dag_reduction[sngl_srcs]).astype(int)
            non_empty_same_ch_mask = ch_subgraph @ ch_subgraph.T
            empty_same_ch_mask = empty_ch_subgraph @ empty_ch_subgraph.T
            same_ch_mask = non_empty_same_ch_mask + empty_same_ch_mask
            np.fill_diagonal(same_ch_mask, 0)
            same_ch_idx = np.argwhere(same_ch_mask == len(self.dag_reduction))
            choices = range(len(same_ch_idx))
            idx = self.rng.choice(choices)
            self.q[move] /= len(choices)
            src_1, src_2 = sngl_srcs[same_ch_idx[idx]]
            chosen_nodes = src_1, src_2
        elif move == "out_del":
            num_max_ancs = self.dag_reduction[sources].sum(0)
            num_pairs = self.n_choose_2(num_max_ancs)
            poss_t = np.flatnonzero(num_pairs)
            p = num_pairs[poss_t] / num_pairs[poss_t].sum()
            t = self.rng.choice(poss_t, p=p)
            self.q[move] *= p[poss_t == t]
            t_max_ancs = np.flatnonzero(self.dag_reduction[sources, t])
            src_1 = sources[self.rng.choice(t_max_ancs)]
            self.q[move] /= len(t_max_ancs)
            if len(t_max_ancs) == 0:
                warnings.warn("inaccurate self.q")
            chosen_nodes = src_1, t
        else:  # then move in ("split", "within", "out_add")
            non_singleton_nodes = np.flatnonzero(self.chain_comps.sum(1) > 1)
            ns_sources = sources[np.in1d(sources, non_singleton_nodes)]
            if move == "split":
                chosen_nodes = self.rng.choice(ns_sources)
                self.q[move] /= len(ns_sources)
            else:  # move in ("within", "out_add")
                # srcs_mask has entry i,j = 1 if and only if
                # i is nonsingleton or
                # i has children other than j's children
                other_children_mask = self.dag_reduction @ ~self.dag_reduction.T
                other_children_mask[ns_sources] = 1
                srcs_mask = other_children_mask[np.ix_(sources, sources)]
                np.fill_diagonal(srcs_mask, 0)
                choices = range(srcs_mask.sum())
                chosen_idx = self.rng.choice(choices)
                self.q[move] /= len(choices)
                chosen_nodes = sources[np.argwhere(srcs_mask)[chosen_idx]]
        return chosen_nodes

    @staticmethod
    def n_choose_2(vec):
        return np.vectorize(lambda n: math.comb(n, 2), otypes=[int])(
            np.array(vec, ndmin=1)
        )

    def expand(self):
        self.cpdag = np.zeros_like(self.cpdag)
        for pa in np.flatnonzero(self.dag_reduction.sum(1)):
            pas = self.chain_comps[pa]
            ch = self.dag_reduction[pa]
            chs = self.chain_comps[ch].sum(0).astype(bool)
            self.cpdag[np.ix_(pas, chs)] = True
        for cc in self.chain_comps:
            nodes = np.flatnonzero(cc)
            for node in nodes:
                ch = nodes[nodes != node]
                self.cpdag[node, ch] = True
        np.fill_diagonal(self.cpdag, False)

    def get_uec(self, cpdag=None):
        if cpdag is None:
            G = self.cpdag
        else:
            G = cpdag
        T = np.eye(len(G)).astype(bool) + G + np.linalg.matrix_power(G, 2)
        recon_uec = T.T @ T
        np.fill_diagonal(recon_uec, False)
        return recon_uec

    def get_likelihood_ratio(self):
        self.expand()
        new_rss = self.compute_mle_rss()
        new_bic = self.num_samps * np.log(new_rss)
        new_bic += self.cpdag.sum() * np.log(self.num_samps)
        # unscaled_ratio = (new_likelihood / self.old_likelihood).astype(np.longdouble)
        return self.old_rss / new_rss, new_rss, new_bic

    def compute_mle_rss(self, graph=None):
        if graph is None:
            graph = self.cpdag
            self.uec = self.get_uec()
        # children_mask = graph.sum(0)
        # children = np.flatnonzero(children_mask)

        # regress = lambda child: lstsq(
        #     self.samples[:, graph[:, child]],
        #     self.samples[:, child],
        #     rcond=None,
        # )[1]
        # rss = float(sum(map(regress, children)))
        # non_children = np.flatnonzero(~children_mask)
        # nc_samps = self.samples[:, non_children]
        # rss += np.diag(nc_samps.T @ nc_samps).sum()
        # # rss here is just n * var(x), which is equiv to the above since data is centered

        # return rss
        return GaussObsL0Pen(self.samples)._mle_full(graph)[1].sum()

    def run_checks(self, move):
        # dag and ccs are correct type
        assert self.chain_comps.dtype == bool
        assert self.dag_reduction.dtype == bool

        # each vertex is a chain component
        assert len(self.chain_comps) == len(self.dag_reduction)
        assert (self.chain_comps.sum(0) == 1).all()

        # every child has a unique parent set with at least two parents
        num_pars = self.dag_reduction.sum(0)
        children = self.dag_reduction[:, num_pars.astype(bool)]
        assert np.unique(children, axis=1).shape[1] == children.shape[1]
        assert (num_pars != 1).all()

        # indeed a DAG and trasitively closed
        num_nodes = len(self.dag_reduction)
        graph = np.copy(self.dag_reduction).astype(int)
        for n in range(2, num_nodes):
            graph += np.linalg.matrix_power(self.dag_reduction, n)
        assert np.diag(graph).sum() == 0
        assert (graph.astype(bool) == self.dag_reduction).all()

        # all edges are essential
        # i.e., all edges are in v-structs
        A = self.dag_reduction
        comp_undir = ~(A + A.T)
        np.fill_diagonal(comp_undir, False)
        assert np.logical_or(~A, comp_undir @ A).all

        # check interesection number
        old_intersection_num = np.logical_not(self.old_dag.sum(0)).sum()
        new_intresection_num = np.logical_not(self.dag_reduction.sum(0)).sum()
        if move == "merge":
            assert old_intersection_num - 1 == new_intresection_num
        elif move == "split":
            assert old_intersection_num + 1 == new_intresection_num
        elif move == "algebraic":
            assert old_intersection_num == new_intresection_num

    def sort(self):
        dag = np.copy(self.dag_reduction) + np.eye(len(self.dag_reduction))
        sorted_idx = np.array([], int)
        while dag.any():
            sink_ccs = np.flatnonzero(dag.sum(1) == 1)
            sinks = np.where(self.chain_comps[sink_ccs])[1]
            sorted_idx = np.append(sinks, sorted_idx)
            dag[sinks] = dag[:, sinks] = 0
        return sorted_idx

    def consider(self, move):
        if move == "merge":
            return self.pick_source_nodes("merge")
        elif move == "split":
            return self.consider_split()
        elif move == "within":
            return self.consider_algebraic("within")
        elif move == "out_del":
            return self.consider_algebraic("out_del")
        elif move == "out_add":
            return self.consider_algebraic("out_add")

    def compute_transition_kernel(self, moves_dict):
        poss_moves = []
        p = []
        considered = {}
        self.q = {key: 1.0 for key in self.p.keys()}
        for move in moves_dict.keys():
            try:
                considered[move] = self.consider(move)
                poss_moves += [move]
                p += [self.p[move]]
            except ValueError:
                continue
        p = np.array(p)
        p /= p.sum()

        return np.array(poss_moves), considered, p

    def scaled_triangle_prior(self, peak_scale):
        if self.pmf is None:
            peak, scale = peak_scale
            min = 0
            max = self.num_feats + 1
            pmf = np.arange(1, max, dtype=float)
            pmf[peak - 1] = 2 / max
            below = pmf[: peak - 1]
            above = pmf[peak:]

            below *= 2 / (max * peak)
            above *= -1
            above += max
            above *= 2 / (max * (max - peak))
            pmf[: peak - 1] = below
            pmf[peak:] = above

            pmf = np.power(pmf, scale)
            pmf /= pmf.sum()
            self.pmf = pmf

        new_num_sources = (self.dag_reduction.sum(0) == 0).sum()
        old_num_sources = (self.old_dag.sum(0) == 0).sum()

        return self.pmf[new_num_sources - 1] / self.pmf[old_num_sources - 1]
