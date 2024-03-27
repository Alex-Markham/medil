from itertools import permutations

import pytest
import numpy as np

from medil.models import MedilCausalModel, NeuroCausalFactorAnalysis


class TestMedilCausalModel:
    def test_sample(self):
        # simple "M" graph, with 2 latent and 3 measurement vars
        biadj = np.zeros((2, 3), bool)
        biadj[[0, 0, 1, 1], [0, 1, 1, 2]] = True
        mcm = MedilCausalModel(biadj)
        mcm.biadj_weights = biadj.astype(float)
        mcm.error_means = np.zeros(3)
        mcm.error_variances = np.ones(3)

        s = mcm.sample(10000)
        assert np.allclose(s.mean(0), mcm.error_means, atol=0.02)

        # when UDG is empty graph
        biadj = np.eye(5, dtype=bool)
        mcm = MedilCausalModel(biadj)
        mcm.biadj_weights = biadj.astype(float)
        mcm.error_means = np.zeros(5)
        mcm.error_variances = np.ones(5)

        dataset = mcm.sample(100000)
        assert np.allclose(dataset.mean(0), mcm.error_means, atol=0.02)

    def test_fit_gauss(self):
        # simple "M" graph, with 2 latent and 3 measurement vars
        biadj = np.zeros((2, 3), bool)
        biadj[[0, 0, 1, 1], [0, 1, 1, 2]] = True
        mcm = MedilCausalModel(biadj)
        mcm.biadj_weights = biadj.astype(float)
        mcm.error_means = np.zeros(3)
        mcm.error_variances = np.ones(3)

        dataset = mcm.sample(10000)

        mcm_est = MedilCausalModel().fit(dataset)
        assert (mcm.biadj == mcm_est.biadj).all()
        assert np.allclose(mcm_est.biadj_weights, mcm.biadj_weights, atol=0.02)
        assert np.allclose(mcm_est.error_means, mcm.error_means, atol=0.02)
        assert np.allclose(mcm_est.error_variances, mcm.error_variances, atol=0.02)

        # randomly generated MCM
        biadj = np.array(
            [
                [False, False, False, True, False],
                [True, True, False, False, False],
                [False, False, True, False, False],
                [True, False, False, False, True],
            ]
        )
        mcm = MedilCausalModel(biadj)
        mcm.biadj_weights = np.array(
            [
                [0.0, 0.0, 0.0, 1.45544253, 0.0],
                [0.90468007, 0.56146029, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.52479145, 0.0, 0.0],
                [1.71990536, 0.0, 0.0, 0.0, 1.86913337],
            ]
        )
        mcm.error_means = np.array(
            [1.87014485, 1.63170711, 0.005477, -1.71480855, -0.06717115]
        )
        mcm.error_variances = np.array(
            [1.31219183, 0.94956784, 1.13403083, 0.54247951, 0.68642491]
        )

        dataset = mcm.sample(10000)

        mcm_est = MedilCausalModel().fit(dataset)

        udg = biadj.T @ biadj
        np.fill_diagonal(udg, False)
        assert (udg == mcm_est.udg).all()

        for p in permutations(range(4)):
            p = np.array(p)
            if (mcm_est.biadj[p] == biadj).all():
                break
        assert (mcm_est.biadj[p] == biadj).all()

        assert np.allclose(mcm_est.biadj_weights[p], mcm.biadj_weights, atol=0.5)

        assert np.allclose(mcm_est.error_means, mcm.error_means, atol=0.05)
        assert np.allclose(mcm_est.error_variances, mcm.error_variances, atol=0.7)


class TestNeuroCausalFactorAnalysis:
    def test_init(self):
        with pytest.raises(NotImplementedError):
            NeuroCausalFactorAnalysis()
