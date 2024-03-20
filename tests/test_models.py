import pytest
import numpy as np

from medil.models import MedilCausalModel, NeuroCausalFactorAnalysis


class TestMedilCausalModel:
    def test_sample(self):
        biadj = np.zeros((2, 3), bool)
        biadj[[0, 0, 1, 1], [0, 1, 1, 2]] = True

        mcm = MedilCausalModel(biadj)
        mcm.biadj_weights = biadj.astype(float)
        mcm.error_means = np.zeros(3)
        mcm.error_variances = np.ones(3)
        s = mcm.sample(100000)
        assert np.allclose(s.mean(0), mcm.error_means, atol=0.02)


class TestNeuroCausalFactorAnalysis:
    def __init__(self):
        with pytest.raises(NotImplementedError):
            NeuroCausalFactorAnalysis()
