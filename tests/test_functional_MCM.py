import warnings

import numpy as np

from medil.functional_MCM import (
    rand_biadj_mat,
    sample_from_minMCM,
    assign_DoF,
    MedilCausalModel,
)
from medil.examples import examples


def test_rand_biadj_mat():
    num_obs = 20
    max_edges = (num_obs * (num_obs - 1)) // 2
    for edge_prob in np.arange(0, 1.1, 0.1):
        biadj_mat = rand_biadj_mat(num_obs, edge_prob)
        assert biadj_mat.sum(1).all()
        udg = (biadj_mat.T @ biadj_mat).astype(bool)
        density = np.triu(udg, 1).sum() / max_edges
        assert np.isclose(edge_prob, density)


def test_sample_from_minMCM():
    biadj_mat = examples[1].MCM.astype(bool)
    _, cov = sample_from_minMCM(biadj_mat)

    obs_cov = cov[3:, :][:, 3:]
    true_indeps = np.array([0, 33, 32, 31, 30, 28])
    test_indeps = np.argsort(np.abs(np.triu(obs_cov, 1).flatten()))
    assert np.in1d(test_indeps[:6], true_indeps).all()

    test_sample, _ = sample_from_minMCM(cov)
    test_cov = np.cov(test_sample, rowvar=False)
    assert np.allclose(test_cov, cov, rtol=0.2, atol=0.3)


def test_assign_DoF():
    biadj_mat = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0]])
    variances = np.array([2.5, 0.33, 2.5, 0.66, 0.88])

    warnings.filterwarnings("error")
    try:
        test_insufficient = assign_DoF(biadj_mat, 2, "uniform")
        assert False
    except UserWarning:
        warnings.resetwarnings()
        warnings.simplefilter("ignore")
        test_insufficient = assign_DoF(biadj_mat, 2, "uniform")
        assert (test_insufficient == biadj_mat).all()

    test_uniform = assign_DoF(biadj_mat, 8, "uniform")
    unique_uniform, counts_uniform = np.unique(test_uniform, axis=0, return_counts=True)
    assert (biadj_mat == unique_uniform).all()
    assert min(counts_uniform) == 2
    assert max(counts_uniform) == 3
    assert counts_uniform.sum() == 8

    test_clique = assign_DoF(biadj_mat, 11, "clique_size")
    unique_clique, counts_clique = np.unique(test_clique, axis=0, return_counts=True)
    assert (biadj_mat == unique_clique).all()
    assert min(counts_clique) == 3
    assert max(counts_clique) == 4
    assert counts_clique.sum() == 11

    test_tot = assign_DoF(biadj_mat, 13, "tot_var", variances)
    unique_tot, counts_tot = np.unique(test_tot, axis=0, return_counts=True)
    assert (biadj_mat == unique_tot).all()
    assert ((5, 2, 6) == counts_tot).all()

    test_avg = assign_DoF(biadj_mat, 29, "avg_var", variances)
    unique_avg, counts_avg = np.unique(test_avg, axis=0, return_counts=True)
    assert (biadj_mat == unique_avg).all()
    assert ((9, 4, 16) == counts_avg).all()

    for dof in range(3, 12):
        for method in ("uniform", "clique_size", "tot_var", "avg_var"):
            test_rounding = assign_DoF(biadj_mat, dof, method, variances)
            assert (np.unique(test_rounding, axis=0) == biadj_mat).all()
            assert dof == len(test_rounding)


def test_rand():
    num_latent, num_obs = 3, 5
    mcm = MedilCausalModel().rand(num_obs, num_latent).biadj_mat
    assert ((num_latent, num_obs) == mcm.shape).all()
