from itertools import permutations

import pytest
import numpy as np

from medil.models import MedilCausalModel, GaussianMCM, NeuroCausalFactorAnalysis


class TestMedilCausalModel:
    def test_base(self):
        mcm = MedilCausalModel()
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

    # def test_ncfa_assign_dof(self):
    #     biadj_mat = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0]])
    #     variances = np.array([2.5, 0.33, 2.5, 0.66, 0.88])

    #     warnings.filterwarnings("error")
    #     try:
    #         test_insufficient = assign_DoF(biadj_mat, 2, "uniform")
    #         assert False
    #     except UserWarning:
    #         warnings.resetwarnings()
    #         warnings.simplefilter("ignore")
    #         test_insufficient = assign_DoF(biadj_mat, 2, "uniform")
    #         assert (test_insufficient == biadj_mat).all()

    #     test_uniform = assign_DoF(biadj_mat, 8, "uniform")
    #     unique_uniform, counts_uniform = np.unique(
    #         test_uniform, axis=0, return_counts=True
    #     )
    #     assert (biadj_mat == unique_uniform).all()
    #     assert min(counts_uniform) == 2
    #     assert max(counts_uniform) == 3
    #     assert counts_uniform.sum() == 8

    #     test_clique = assign_DoF(biadj_mat, 11, "clique_size")
    #     unique_clique, counts_clique = np.unique(
    #         test_clique, axis=0, return_counts=True
    #     )
    #     assert (biadj_mat == unique_clique).all()
    #     assert min(counts_clique) == 3
    #     assert max(counts_clique) == 4
    #     assert counts_clique.sum() == 11

    #     test_tot = assign_DoF(biadj_mat, 13, "tot_var", variances)
    #     unique_tot, counts_tot = np.unique(test_tot, axis=0, return_counts=True)
    #     assert (biadj_mat == unique_tot).all()
    #     assert ((5, 2, 6) == counts_tot).all()

    #     test_avg = assign_DoF(biadj_mat, 29, "avg_var", variances)
    #     unique_avg, counts_avg = np.unique(test_avg, axis=0, return_counts=True)
    #     assert (biadj_mat == unique_avg).all()
    #     assert ((9, 4, 16) == counts_avg).all()

    #     for dof in range(3, 12):
    #         for method in ("uniform", "clique_size", "tot_var", "avg_var"):
    #             test_rounding = assign_DoF(biadj_mat, dof, method, variances)
    #             assert (np.unique(test_rounding, axis=0) == biadj_mat).all()
    #             assert dof == len(test_rounding)


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

        dataset = mcm.sample(10000)

        NeuroCausalFactorAnalysis(verbose=True).fit(dataset)
