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

        s = mcm.sample(100000)
        assert np.allclose(s.mean(0), mcm.error_means, atol=0.02)

    def test_fit_gauss(self):
        biadj = True


class TestNeuroCausalFactorAnalysis:
    def test_init(self):
        with pytest.raises(NotImplementedError):
            NeuroCausalFactorAnalysis()
