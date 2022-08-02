"""Implement the Greedy Unconditional Equivalence Search (GUES) algorithm."""
import numpy as np
from .gauss_obs_l0_pen import GaussObsL0Pen
from numpy.linalg import lstsq, norm
from scipy.stats import chi2, beta
from scipy.spatial.distance import pdist, squareform


class InputData(object):
    r"""Feed data into GUES to learn its Markov equivalence class.

    Attributes
    ----------
    samples : array_like
        Passes directly to numpy.array to get a :math:`M \times N`
        matrix with :math:`N` samples of :math:`M` random variables.

    Methods
    -------
    gues_MEC(test="gauss", alpha=0.05)
        Return optimal essential graph (representing a Markov
        equivalence class of DAGs) within unconditional equivalence
        class.

    estimate_UEC(test="gauss", alpha=0.05)
        Return undirected graph resulting from unconditional
        independence tests.

    initialize_MEC()
        Return the maximal essential graph in the unconditional
        equivalence class.

    """

    def __init__(self, samples):
        self.samples = np.array(samples, dtype=float)
        self.num_samps, self.num_feats = self.samples.shape

    def gues_MEC(self, test="gauss", alpha=0.01, score_func="GOL0"):
        r"""Return optimal essential graph (representing a Markov
        equivalence class of DAGs) within unconditional equivalence
        class.

        Notes
        -----
        Compare to pseudocode in Alg 3 of paper.
        """
        self.score_func = score_func
        self.estimate_uec(test, alpha)
        self.init_CPDAG()
        self.score_obj = GaussObsL0Pen(self.samples)
        self.score = self.score_obj.full_score(self.cpdag)
        while True:
            # get possible moves
            # find best scoring move and its score
            if best_score <= 0:
                break
            # take best move
            # local update to score

    def estimate_UEC(self, test, alpha):
        r"""Return undirected graph resulting from independence tests."""
        if test == "gauss":
            corr = np.corrcoef(self.samples, rowvar=False)
            dist = beta(self.num_samps / 2 - 1, self.num_samps / 2 - 1, loc=-1, scale=2)
            crit_val = abs(dist.ppf(alpha / 2))
            uec = abs(corr) >= crit_val

        if test == "dcov_fast":
            cov, d_bars = dcov(self.samples)
            crit_val = chi2(1).ppf(1 - alpha)
            test_val = self.num_samps * cov / np.outer(d_bars, d_bars)
            uec = test_val >= crit_val

        # if test == "dcov_accurate":
        #     dcov =
        #     p_vals =            # perm test
        #     uec = p_vals[p_vals<=alpha]

        if test == "algebraic":
            # add submodule for this
            pass

        self.uec = uec

    def init_cpdag(self):
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

        # add test to see if num connected components in essential
        # graph is less than in UEC, indicating data isn't from a DAG?

    def rmable_edges(self):
        # undir_mask[i, j] == 1 if and only if i -- j rmable in cpdag
        # i.e., if and only if they have the same parents or are in a
        # k>3-clique
        undir_mask = np.logical_and((self.cpdag @ self.cpdag).T, self.cpdag)
        undir_edges = np.argwhere(np.triu(undir_mask))

        dir_component = np.logical_and(
            self.cpdag, np.logical_xor(self.cpdag, self.cpdag.T)
        )
        min_ants = self.get_min_ants()
        poss_rmable = np.argwhere(dir_component)
        other_pars_w = dir_component.T[poss_rmable[:, 1]]
        other_pars_w[np.arange(len(other_pars_w)), poss_rmable[:, 0]] = 0
        other_min_ants_w = other_pars_w @ min_ants
        min_ants_v = min_ants[poss_rmable[:, 0]]
        rmable = np.logical_or(np.logical_not(min_ants_v), other_min_ants_w).all(1)
        # dir_edges[a] = v_w == [i, j] if and only if min ants of i are contained in
        # min ants of pa(j)\{i}
        dir_edges = poss_rmable[rmable]

        return undir_edges, dir_edges

    def get_min_ants(self):
        ## find min anteriors:
        # reduce all chain components to one node; cpdag becomes a dag
        reduction, chain_comps = self.chain_reduction(
            np.copy(self.cpdag), np.eye(self.num_feats)
        )

        # sort and find transitive closure of reduced dag
        num_nodes = len(reduction)
        sorted_idx = self.topological_sort(reduction)
        sorted_reduction = reduction[sorted_idx][:, sorted_idx]
        dag_eye = sorted_reduction + np.eye(num_nodes)
        trans_closure = (
            np.linalg.matrix_power(dag_eye, np.ceil(np.log2(num_nodes)).astype(int))
            - np.eye(num_nodes)
        ).astype(bool)

        # find max ancestors
        source_mask = np.logical_not(trans_closure.sum(axis=0))
        sorted_max_ancs = np.zeros_like(reduction)
        sorted_max_ancs[source_mask] = (
            trans_closure[source_mask] + np.eye(num_nodes)[source_mask]
        )
        inv_order = np.argsort(sorted_idx)
        # max_ancs[i, j] == 1 if and only if j is a max anc of i
        max_ancs = sorted_max_ancs[inv_order][:, inv_order].T

        # use chain components and max ancs to get min anteriors
        # min_ants[i, j] == 1 if and only j is min ant of i
        min_ants = np.zeros_like(self.cpdag)
        for idx, comp in enumerate(chain_comps):
            min_ants[comp] = chain_comps[max_ancs[idx].astype(bool)].sum(0)
        return min_ants

    @staticmethod
    def chain_reduction(cpdag, chain_components):
        # caution! pdag gets changed in place, so make copy first
        undir_component = cpdag * cpdag.T
        if undir_component.any():
            v, w = np.unravel_index(undir_component.argmax(), undir_component.shape)

            cpdag[w] += cpdag[v]
            cpdag[:, w] += cpdag[:, v]
            cpdag[w, w] = 0
            r_cpdag = np.delete(cpdag, v, 0)
            r_cpdag = np.delete(r_cpdag, v, 1)

            chain_components[w] += chain_components[v]
            r_ccs = np.delete(chain_components, v, 0)

            return InputData.chain_reduction(r_cpdag, r_ccs)
        else:
            return cpdag, chain_components.astype(bool)

    @staticmethod
    def topological_sort(dag):
        dag = np.copy(dag) + np.eye(len(dag))
        sorted_idx = np.array([], int)
        while dag.any():
            sinks = np.flatnonzero(dag.sum(1) == 1)
            sorted_idx = np.append(sinks, sorted_idx)
            dag[sinks] = dag[:, sinks] = 0
        return sorted_idx

    # maybe it's best to construct mt set in init_cpdag(), and then
    # update it each time an edge is removed?

    # Also think about if it's possible to score a PDAG and if that
    # could be used to deduce which PDAG contains the best-scoring
    # width-1 completion

    def score_undir_rm(self):
        return

    def score_dir_rm(self):
        return

    def get_T(self, rmed_edge):
        v, w = rmed_edge
        ne_v = np.logical_or(self.cpdag[v], self.cpdag[:, v])
        ne_cc_w = np.logical_and(self.cpdag[w], self.cpdag[:, w])
        return np.logical_and(ne_v, ne_cc_w)

    def undir_completions(self, rmed_edge):
        v, w = rmed_edge
        T_mask = get_T(rmed_edge)
        num_T = T_mask.sum()
        if num_T > 1:
            # remove undirected edge
            rmed_cpdag = np.copy(self.cpdag)
            rmed_cpdag[v, w] = rmed_cpdag[w, v] = 0

            # for t in T, direct all v -- t' -- w into v-structures, with a new cpdag for each
            T = np.flatnonzero(T_mask)
            cpdags = np.tile(self.rmed_cpdag[np.newaxis], num_T, axis=0)
            for offset in range(1, num_T):
                offset_T = np.roll(T, offset)
                cpdags[:, offset_T, v] = cpdags[:, offset_T, w] = 0
            return cpdags
        else:  # then already  a CPDAG
            cpdag = np.copy(self.cpdag)
            cpdag[v, w] = cpdag[w, v] = 0
            return np.expand_dims(cpdag, axis=0)

    def dir_completions(self, rmed_edge):
        v, w = rmed_edge
        T = self.get_T(rmed_edge)

        return cpdags


def dcov(samples):
    r"""Compute sample distance covariance matrix.

    Parameters
    ----------
    samples : 2d numpy array of floats
              A :math:`N \times M` matrix with :math:`N` samples of
              :math:`M` random variables.

    Returns
    -------
    cov : 2d numpy array
          A square matrix :math:`C`, where :math:`C_{i,j}` is the
          sample distance covariance between random variables
          :math:`R_i` and :math:`R_j`.
    d_bars : 1d numpy array
             The means of the distance matrixes for each feature, used
             in independence testing.

    """
    num_samps, num_feats = samples.shape
    dists = np.zeros((num_feats, num_samps**2))
    d_bars = np.zeros(num_feats)
    # compute doubly centered distance matrix for every feature:
    for feat_idx in range(num_feats):
        n = num_samps
        t = np.tile
        # raw distance matrix:
        d = squareform(pdist(samples[:, feat_idx].reshape(-1, 1), "cityblock"))
        # doubly centered:
        d_bar = d.mean()
        d -= t(d.mean(0), (n, 1)) + t(d.mean(1), (n, 1)).T - t(d_bar, (n, n))
        dd = d.flatten()
        dists[feat_idx] = dd / n
        d_bars[feat_idx] = d_bar
    cov = dists @ dists.T
    return cov, d_bars


def gen_samples(num_nodes, p, num_samples, rng, use_sempler=False):
    row_idx, col_idx = np.triu_indices(num_nodes, k=1)
    edge_mask = rng.choice((True, False), size=len(row_idx), p=(p, 1 - p))

    dag = np.zeros((num_nodes, num_nodes), bool)
    dag[row_idx, col_idx] = edge_mask

    num_edges = edge_mask.sum()
    weights = np.zeros_like(dag, float)
    weights[dag] = (rng.random(num_edges) * 2) * (
        rng.integers(2, size=num_edges) * 2 - 1
    )

    if use_sempler:
        means = np.zeros(num_nodes)
        variances = np.ones(num_nodes)
        lganm = LGANM(weights, means, variances)
        samples = lganm.sample(num_samples)
    else:
        means = 0  # (rng.random() * 4) - 2
        st_dvs = 1  # rng.random() * 2
        samples = rng.normal(means, st_dvs, (num_samples, num_nodes))

        for feature, parents in zip(samples.T, dag.T):
            feature += samples @ parents
    order = rng.permutation(num_nodes)
    samples = samples[:, order]

    return dag, order, samples


def norm_hamming_sim(g, h):
    num_nodes = len(g)
    num_poss_edges = (num_nodes * (num_nodes - 1)) / 2
    sim = (g + g.T == h + h.T).sum() - num_nodes
    return sim / (2 * num_poss_edges)


def struct_hamming_sim(g, h):
    num_nodes = len(g)
    num_poss_edges = num_nodes**2 - num_nodes
    sim = (g == h).sum() - num_nodes
    return sim / (num_poss_edges)
