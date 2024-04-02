import pytest
import numpy as np

from medil.sample import mcm, biadj


def test_biadg():
    b = biadj(num_meas=5, num_latent=5)
    assert (b.sum(0) == 1).all()
    assert (b.sum(1) == 1).all()

    b = biadj(num_meas=5, density=0)
    assert (b.sum(0) == 1).all()
    assert b.sum(1).all()

    with pytest.raises(ValueError):
        biadj(num_meas=5, num_latent=1, one_pure_child=False)

    b = biadj(num_meas=10, one_pure_child=False)
    assert b.sum(0).all()
    assert b.sum(1).all()

    num_meas = 20
    max_edges = (num_meas * (num_meas - 1)) // 2
    for density in np.arange(0, 1.1, 0.1):
        biadj_mat = biadj(num_meas, density, one_pure_child=False)
        assert biadj_mat.sum(1).all()
        udg = (biadj_mat.T @ biadj_mat).astype(bool)
        density = np.triu(udg, 1).sum() / max_edges
        assert np.isclose(density, density)


def test_mcm():
    with pytest.raises(NotImplementedError):
        mcm(num_meas=2, parameterization="VAE")

    with pytest.raises(ValueError):
        mcm(num_meas=2, parameterization="test")

    m = mcm(num_meas=5)
    params = m.parameters
    assert params.biadj_weights[m.biadj].all()
    assert ~params.biadj_weights[~m.biadj].any()
    assert params.error_variances.all()
    assert params.error_means.all()
