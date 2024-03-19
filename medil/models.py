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
        udg_method: str = "bic",
        rng=default_rng(0),
    ) -> None:
        self.biadj = biadj
        self.udg = udg
        self.parameterization = parameterization
        self.one_pure_child = one_pure_child
        self.udg_method = udg_method
        self.rng = rng
        if parameterization == "gauss":
            self.biadj_weights = None
            self.error_means = None
            self.error_variances = None
            # self.params = {biadj_weights: None, error_means: None,
            # error_variances: None}
        elif parameterization == "vae":
            self.vae = None

    def fit(self, dataset: npt.NDArray) -> "MedilCausalModel":
        """"""
        self.dataset = dataset
        if self.biadj is None:
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
        if self.parameterization == "vae":
            # samp = sample drawn from vae model
            print("not implemented yet :(")
            sample = None
        elif self.parameterization == "gauss":
            num_latent, num_meas = self.biadj.shape
            latent_sample = self.rng.multivariate_normal(
                np.zeros(num_latent), np.eye(num_latent), sample_size
            )
            error_sample = self.rng.multivariate_normal(
                self.error_means, self.error_variances, sample_size
            )
            sample = self.biadj_weights.T @ latent_sample + error_sample
        return sample


class NeuroCausalFactorAnalysis(MedilCausalModel):
    pass
