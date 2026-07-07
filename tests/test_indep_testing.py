import numpy as np
import pytest
from numpy.random import default_rng

from medil.independence_testing import dcov, estimate_UDG, xicorr


def test_xicorr_monotone():
    """Monotone relationship should produce positive xi and small p-value."""
    x = list(range(1, 21))
    y = [v**2 for v in x]
    xi, pvalue = xicorr(x, y)
    assert xi > 0
    assert pvalue < 0.05


def test_xicorr_size_mismatch():
    with pytest.raises(ValueError):
        xicorr([1, 2, 3], [1, 2])


def test_dcov_independent():
    """Independent Gaussians should have near-zero distance covariance."""
    rng = default_rng(0)
    n = 500
    x = rng.standard_normal((n, 2))
    cov, _ = dcov(x)
    assert abs(cov[0, 1]) < 0.05


def test_dcov_dependent():
    """Linearly dependent variables should have nonzero distance covariance."""
    rng = default_rng(0)
    n = 500
    z = rng.standard_normal(n)
    x = np.column_stack([z, z + 0.1 * rng.standard_normal(n)])
    cov, _ = dcov(x)
    assert cov[0, 1] > 0.1


def test_estimate_UDG_dcov_fast():
    """Simple M-graph: UDG should show dependence between the shared-parent pair."""
    rng = default_rng(0)
    biadj = np.zeros((2, 3), bool)
    biadj[[0, 0, 1, 1], [0, 1, 1, 2]] = True
    weights = biadj.astype(float)
    latent = rng.multivariate_normal(np.zeros(2), np.eye(2), 1000)
    errors = rng.multivariate_normal(np.zeros(3), np.eye(3), 1000)
    dataset = latent @ weights + errors

    udg, _ = estimate_UDG(dataset, method="dcov_fast")
    np.fill_diagonal(udg, False)

    expected_udg = biadj.T @ biadj
    np.fill_diagonal(expected_udg, False)
    assert (udg == expected_udg).all()


def test_estimate_UDG_xicor():
    """Same M-graph, using xi correlation method."""
    rng = default_rng(0)
    biadj = np.zeros((2, 3), bool)
    biadj[[0, 0, 1, 1], [0, 1, 1, 2]] = True
    weights = biadj.astype(float)
    latent = rng.multivariate_normal(np.zeros(2), np.eye(2), 1000)
    errors = rng.multivariate_normal(np.zeros(3), np.eye(3), 1000)
    dataset = latent @ weights + errors

    udg, _ = estimate_UDG(dataset, method="xicor")
    np.fill_diagonal(udg, False)

    expected_udg = biadj.T @ biadj
    np.fill_diagonal(expected_udg, False)
    assert (udg == expected_udg).all()


def test_estimate_UDG_gtest_not_implemented():
    rng = default_rng(0)
    data = rng.standard_normal((100, 3))
    with pytest.raises(NotImplementedError):
        estimate_UDG(data, method="g-test")
