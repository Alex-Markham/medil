"""MeDIL causal model base class and a preconfigured NCFA class."""
import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from scipy.optimize import minimize

from .ecc_algorithms import find_heuristic_1pc


class MedilCausalModel(object):
    def __init__(
        self,
        biadj: None | npt.NDArray = None,
        udg: None | npt.NDArray = None,
        parameterization: str = "gauss",
        one_pure_child: bool = True,
        udg_method: str = "default",
        rng=default_rng(0),
    ) -> None:
        self.biadj = biadj
        self.udg = udg
        self.parameterization = parameterization
        self.one_pure_child = one_pure_child
        self.udg_method = udg_method
        self.rng = rng
        if parameterization == "gauss":
            if self.udg_method == "default":
                self.udg_method = "bic"
            self.biadj_weights = None
            self.error_means = None
            self.error_variances = None
            # self.params = {biadj_weights: None, error_means: None,
            # error_variances: None}
        elif parameterization == "vae":
            if self.udg_method == "default":
                self.udg_method = "xicor"
            self.vae = None

    def fit(self, dataset: npt.NDArray) -> "MedilCausalModel":
        """"""
        self.dataset = dataset
        if self.biadj is None:
            self._compute_biadj()

        if self.parameterization == "gauss":
            self.error_means = self.dataset.mean(0)
            cov = np.cov(self.dataset, rowvar=False)

            num_weights = self.biadj.sum()
            num_err_vars = self.biadj.shape[1]

            def _objective(weights_and_err_vars):
                weights = weights_and_err_vars[:num_weights]
                err_vars = weights_and_err_vars[num_weights:]

                biadj_weights = np.zeros_like(self.biadj, float)
                biadj_weights[self.biadj] = weights

                return (
                    cov - biadj_weights.T @ biadj_weights - np.diagflat(err_vars)
                ) ** 2

            vectorized_solution = minimize(
                _objective, np.ones(num_weights + num_err_vars)
            )
            weights = vectorized_solution[:num_weights]
            self.error_variances = vectorized_solution[num_weights:]
            self.biadj_weights = np.zeros_like(self.biadj, float)
            self.biadj_weights[self.biadj] = weights
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
        if self.udg_method == "bic":
            samp_size = len(self.dataset)
            cov = np.cov(self.dataset, rowvar=False)
            corr = np.corrcoef(self.dataset, rowvar=False)
            inner_numerator = 1 - cov * corr  # should never be <= 0?
            inner_numerator = inner_numerator.clip(min=0.00001)
            inner_numerator[np.tril_indices_from(inner_numerator)] = 1
            udg_triu = np.log(inner_numerator) < (-np.log(samp_size) / samp_size)
            udg = udg_triu + udg_triu.T
        else:
            num_meas = self.dataset.shape[1]
            udg = np.ones((num_meas, num_meas), bool)
        self.udg = udg

    def sample(self, sample_size: int) -> npt.NDArray:
        if self.parameterization == "vae":
            # samp = sample drawn from vae model
            print("not implemented yet :(")
            sample = None
        elif self.parameterization == "gauss":
            num_latent = len(self.biadj)
            latent_sample = self.rng.multivariate_normal(
                np.zeros(num_latent), np.eye(num_latent), sample_size
            )
            error_sample = self.rng.multivariate_normal(
                self.error_means, np.diagflat(self.error_variances), sample_size
            )
            sample = latent_sample @ self.biadj_weights + error_sample
        return sample


class NeuroCausalFactorAnalysis(MedilCausalModel):
    def __init__(self):
        raise (NotImplementedError)
