"""MeDIL causal model base class and a preconfigured NCFA class."""
import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

from .ecc_algorithms import find_heuristic_1pc


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

    def fit(self, dataset: npt.NDArray) -> "MedilCausalModel":
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

        return self

    def _compute_biadj(self):
        if self.udg is None:
            self._estimate_udg()
        self.biadj = find_heuristic_1pc(self.udg)

    def _estimate_udg(self):
        if self.parameterization == "gauss":
            udg = bic_optimal()
        elif self.parameterization == "vae":
            udg = True

    def sample(self, sample_size: int) -> npt.NDArray:
        if hasattr(self, "vae"):
            # samp = sample drawn from vae model
            pass
        else:
            if not hasattr(self, "cov"):
                # generate random weights in +-[0.5, 2]
                num_edges = self.biadj.sum()
                idcs = np.argwhere(self.biadj)
                idcs[:, 1] += self.num_latent

                weights = (self.rng.random(num_edges) * 1.5) + 0.5
                weights[self.rng.choice((True, False), num_edges)] *= -1

                precision = np.eye(self.num_latent + self.num_obs, dtype=float)
                precision[idcs[:, 0], idcs[:, 1]] = weights
                precision = precision.dot(precision.T)

                cov = np.linalg.inv(precision)
                self.cov = cov

            samp = self.rng.multivariate_normal(
                np.zeros(len(self.cov)), self.cov, sample_size
            )
        return samp


class NeuroCausalFactorAnalysis(MedilCausalModel):
    pass
