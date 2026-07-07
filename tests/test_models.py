from itertools import permutations

import numpy as np
import pytest
import torch
from medil.models import (
    GaussianMCM,
    _MedilCausalModel,
    NeuroCausalFactorAnalysis,
)


class TestMedilCausalModel:
    def test_base(self):
        mcm = _MedilCausalModel()
        with pytest.raises(NotImplementedError):
            mcm.fit(np.array([]))
        with pytest.raises(NotImplementedError):
            mcm.sample(0)


class TestGaussianMCM:
    def test_sample_m(self):
        """Simple "M" graph, with 2 latent and 3 measurement vars."""
        biadj = np.zeros((2, 3), bool)
        biadj[[0, 0, 1, 1], [0, 1, 1, 2]] = True
        mcm = GaussianMCM(biadj=biadj)
        params = mcm.parameters
        params.biadj_weights = biadj.astype(float)
        params.error_means = np.zeros(3)
        params.error_variances = np.ones(3)

        s = mcm.sample(10000)
        print(s)
        assert np.allclose(s.mean(0), mcm.parameters.error_means, atol=0.02)

    def test_sample_empty(self):
        """When UDG is empty graph."""
        biadj = np.eye(5, dtype=bool)
        mcm = GaussianMCM(biadj=biadj)
        params = mcm.parameters
        params.biadj_weights = biadj.astype(float)
        params.error_means = np.zeros(5)
        params.error_variances = np.ones(5)

        dataset = mcm.sample(100000)
        assert np.allclose(dataset.mean(0), mcm.parameters.error_means, atol=0.02)

    def test_fit_m(self):
        """Simple "M" graph, with 2 latent and 3 measurement vars."""
        biadj = np.zeros((2, 3), bool)
        biadj[[0, 0, 1, 1], [0, 1, 1, 2]] = True
        mcm = GaussianMCM(biadj=biadj)
        params = mcm.parameters
        params.biadj_weights = biadj.astype(float)
        params.error_means = np.zeros(3)
        params.error_variances = np.ones(3)

        dataset = mcm.sample(10000)

        mcm_est = GaussianMCM().fit(dataset)
        params_est = mcm_est.parameters
        assert (mcm.biadj == mcm_est.biadj).all()
        assert np.allclose(params_est.biadj_weights, params.biadj_weights, atol=0.02)
        assert np.allclose(params_est.error_means, params.error_means, atol=0.02)
        assert np.allclose(
            params_est.error_variances, params.error_variances, atol=0.02
        )

    def test_fit_random(self):
        """Randomly generated MCM."""
        biadj = np.array(
            [
                [False, False, False, True, False],
                [True, True, False, False, False],
                [False, False, True, False, False],
                [True, False, False, False, True],
            ]
        )
        mcm = GaussianMCM(biadj=biadj)
        params = mcm.parameters
        params.biadj_weights = np.array(
            [
                [0.0, 0.0, 0.0, 1.45544253, 0.0],
                [0.90468007, 0.56146029, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.52479145, 0.0, 0.0],
                [1.71990536, 0.0, 0.0, 0.0, 1.86913337],
            ]
        )
        params.error_means = np.array(
            [1.87014485, 1.63170711, 0.005477, -1.71480855, -0.06717115]
        )
        params.error_variances = np.array(
            [1.31219183, 0.94956784, 1.13403083, 0.54247951, 0.68642491]
        )

        dataset = mcm.sample(10000)

        mcm_est = GaussianMCM().fit(dataset)
        params_est = mcm_est.parameters

        udg = biadj.T @ biadj
        np.fill_diagonal(udg, False)
        assert (udg == mcm_est.udg).all()

        for p in permutations(range(4)):
            p = np.array(p)
            if (mcm_est.biadj[p] == biadj).all():
                break
        assert (mcm_est.biadj[p] == biadj).all()

        assert np.allclose(params_est.biadj_weights[p], params.biadj_weights, atol=0.5)

        assert np.allclose(params_est.error_means, params.error_means, atol=0.05)
        assert np.allclose(params_est.error_variances, params.error_variances, atol=0.7)


class TestNeuroCausalFactorAnalysis:
    def test_fit_m_gaussian(self):
        """Simple "M" graph, with 2 latent and 3 measurement vars, sampled from GaussianMCM."""
        biadj = np.zeros((2, 3), bool)
        biadj[[0, 0, 1, 1], [0, 1, 1, 2]] = True
        mcm = GaussianMCM(biadj=biadj)
        params = mcm.parameters
        params.biadj_weights = biadj.astype(float)
        params.error_means = np.zeros(3)
        params.error_variances = np.ones(3)

        dataset = mcm.sample(2000)

        # standardize
        dataset -= dataset.mean(0)
        dataset /= dataset.std(0)

        ncfa = NeuroCausalFactorAnalysis(verbose=False)
        ncfa.hyperparams.update(
            {
                "mu": 0.01,
                "lambda": 0.01,
                "deg_of_free": 5,
                "width_per_meas": 5,
                "num_hidden_layers": 1,
                "num_epochs": 200,
                "lr": 0.01,
            }
        )
        ncfa.fit(dataset)

        d = torch.Tensor(dataset[:5])
        recon_d = ncfa.parameters.vae(d)[0]

        ncfa = NeuroCausalFactorAnalysis(verbose=False)
        ncfa.hyperparams.update(
            {
                "mu": 0.0,
                "lambda": 0,  # 0.015,
                "deg_of_free": 5,
                "width_per_meas": 5,
                "num_hidden_layers": 1,
                "num_epochs": 200,
                "lr": 0.01,
            }
        )
        ncfa.fit(dataset)

        torch.Tensor(dataset[:5])  # d
        ncfa.parameters.vae(torch.Tensor(dataset[:5]))[0]  # recon_d

    def test_sample_m_gaussian(self):
        """sample() returns correct shape after fitting an M-graph."""
        biadj = np.zeros((2, 3), bool)
        biadj[[0, 0, 1, 1], [0, 1, 1, 2]] = True
        mcm = GaussianMCM(biadj=biadj)
        mcm.parameters.biadj_weights = biadj.astype(float)
        mcm.parameters.error_means = np.zeros(3)
        mcm.parameters.error_variances = np.ones(3)
        dataset = mcm.sample(2000)
        dataset = (dataset - dataset.mean(0)) / dataset.std(0)

        ncfa = NeuroCausalFactorAnalysis(biadj=biadj, verbose=False)
        ncfa.hyperparams["num_epochs"] = 5
        ncfa.fit(dataset)

        out = ncfa.sample(50)
        assert out.shape == (50, 3)

        out, latent = ncfa.sample(50, include_latent=True)
        assert out.shape == (50, 3)
        assert latent.shape[0] == 50
