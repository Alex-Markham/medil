"""MeDIL causal model base class and a preconfigured NCFA class."""
import numpy as np
import numpy.typing as npt


class MedilCausalModel(object):
    def __init__(
        self,
        biadj: None | npt.NDArray = None,
        udg: None | npt.NDArray = None,
        parameterization: str = "gauss",
        one_pure_child: bool = True,
        udg_method: str = "constraint-based",
        rng=default_rng(0),
    ) -> None:
        self.biadj = biadj
        self.udg = udg
        self.parameterization = parameterization
        self.one_pure_child = one_pure_child
        self.udg_method = udg_method
        self.rng = rng

    def fit(self, dataset: npt.NDArray) -> MedilCausalModel:
        """"""

        self.dataset = dataset
        if self.biadj_mat is None:
            self._compute_biadj()

        if self.parameterization == "gauss":
            # either use scipy minimize or implement gradient descent
            # myself in numpy, or try to find more info about/how to
            # implement MLE (check MLE in Factor analysis---an
            # algebraic derivation by stoica and jansson)
            pass

        pass

    def _compute_biadj(self):
        if self.udg is None:
            self._estimate_udg()
        self.biadj = find_heuristic_1pc(self.udg)

    def _estimate_udg(self):
        if self.paramaterization == "gauss":
            udg = bic_optimal()
        elif self.paramaterization == "vae":
            udg = True

    def sample(self, sample_size: int) -> npt.NDArray:
        if hasattr(self, "vae"):
            # samp = sample drawn from vae model
            pass
        else:
            if not hasattr(self, "cov"):
                # generate random weights in +-[0.5, 2]
                num_edges = self.biadj_mat.sum()
                idcs = np.argwhere(self.biadj_mat)
                idcs[:, 1] += self.num_latent

                weights = (self.rng.random(num_edges) * 1.5) + 0.5
                weights[self.rng.choice((True, False), num_edges)] *= -1

                precision = np.eye(self.num_latent + self.num_obs, dtype=float)
                precision[idcs[:, 0], idcs[:, 1]] = weights
                precision = precision.dot(precision.T)

                cov = np.linalg.inv(precision)
                self.cov = cov

            samp = sef.rng.multivariate_normal(
                np.zeros(len(self.cov)), self.cov, sample_size
            )
        return samp

    def rand_struct(self, num_obs, num_latent=None, edge_prob=None):
        self.num_obs = num_obs

        if num_latent is not None and edge_prob is not None:
            raise ValueError(
                "You may specify `num_latent` or `edge_prob` but not both."
            )
        elif num_latent is not None:
            self.num_latent = num_latent
            self.rand_grues()
        else:
            if edge_prob is None:
                edge_prob = 0.5
            self.rand_er(edge_prob)

        return self

    def rand_params(self):
        return False

    def rand_er(self, edge_prob):
        """Generate minMCM from Erdős–Rényi random undirected graph
        over observed variables."""
        # ER random graph
        udg = np.zeros((self.num_obs, self.num_obs), bool)
        max_edges = (self.num_obs * (self.num_obs - 1)) // 2
        num_edges = np.round(edge_prob * max_edges).astype(int)
        edges = np.ones(max_edges)
        edges[num_edges:] = 0
        udg[np.triu_indices(self.num_obs, k=1)] = self.rng.permutation(edges)
        udg += udg.T
        np.fill_diagonal(udg, True)
        self.udg = udg

        # find latent connections (minimum edge clique cover)
        biadj_mat = find_cm(udg)
        self.biadj_mat = biadj_mat.astype(bool)
        self.num_latent = len(biadj_mat)

    # if we have 1pc, then we take num M, num L<M, and a density param

    # otherwise, do the above in rand_er?


class NeuroCausalFactorAnalysis(MedilCausalModel):
    pass
