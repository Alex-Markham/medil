"""Implements constraint-based graphical causal abstraction algorithms."""
import numpy as np
from .independence_testing import estimate_UDG


class MicroData(object):
    r"""Represents microlevel data from which abstractions can be learned.

    Attributes
    ----------
    samples : PyDictObject
        Dictionary in which the first item is an observational data
        set and all other items are interventional data sets. Each
        item assumed to be a :math:`N \times M` numpy.array with
        :math:`N` samples of :math:`M` random variables. `N` may vary
        between items but `M` may not.

    Methods
    -------
    marginal_abstraction()
        Return macrolevel I-DAG using microlevel pairwise marginal
        independence constraints.

    single_target_abstraction()
        Return macrolevel I-CPDAG where each intervention is on a
        single target.

    """

    def __init__(self, samples, interventions):
        self.samples = samples

        keys = list(samples.keys())
        self.obs = keys[0]
        self.intervs = keys[1:]

        self.obs_samps = samples[self.obs]

        self.interventions = interventions
        # needs to be edge list for E_I and E_{I,V}; np.argwhere form?

    # needs arg for interv PO; easier than learning from data/assuming
    # total order
    def marginal_abstraction(self, pmi_cons=None, pii_cons=None):
        # initialize empty macro I-CPDAG
        num_macro_caus_vars = np.shape(self.obs_samps)[1]
        num_macro_interv_vars = len(self.intervs)
        num_macro_vars = num_macro_caus_vars + num_macro_interv_vars
        macro_ICPDAG = np.zeros((num_macro_vars, num_macro_vars), bool)

        # add intervention nodes and edges to macro I-CPDAG
        macro_ICPDAG[self.interventions] = True

        if pmi_cons is None:
            # estimate pairwise marginal independence constraints to
            # get super-skeleton of macro-CPDAG
            macro_CPDAG = estimate_UDG(self.obs_samps)
        else:
            macro_CPDAG = np.ones((num_macro_caus_vars, num_macro_caus_vars), bool)
            macro_CPDAG[pmi_cons] = False
            macro_CPDAG = macro_CPDAG.T & macro_CPDAG
            np.fill_diagonal(macro_CPDAG, False)

        # get CPDAG; see medil.grues.get_max_cpdag()
        compliment_U = ~macro_CPDAG
        np.fill_diagonal(compliment_U, False)
        V = compliment_U @ macro_CPDAG
        W = np.logical_and(V, macro_CPDAG).T
        macro_CPDAG[W] = False

        # fill in edges between causal variables
        macro_ICPDAG[np.argwhere(macro_CPDAG)] = True

        # now need to use pii_cons to complete the existing
        # macro_IPDAG
        if pii_cons is None:
            pii_cons = self.get_pii_constraints()

        # pii_cons are what don't change;
        # figure out how that determines some PEOs

        # then collapse chain comps using grues code
        micro_ICPDAG = np.copy(macro_ICPDAG)
        undir = np.logical_and(micro_ICPDAG, micro_ICPDAG.T)
        chain_comps = np.eye(num_macro_caus_vars).astype(bool)
        while undir.any() and len(undir) > 1:
            v, w = np.unravel_index(undir.argmax(), undir.shape)
            cpdag = np.delete(micro_ICPDAG, v, 0)
            cpdag = np.delete(micro_ICPDAG, v, 1)
            undir = np.delete(undir, v, 0)
            undir = np.delete(undir, v, 1)
            chain_comps[w] += chain_comps[v]
            chain_comps = np.delete(chain_comps, v, 0)

        return micro_ICPDAG, chain_comps

    def get_pii_constraints(self):
        return

    def single_target_abstraction():
        return
