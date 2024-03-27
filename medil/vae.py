def assign_DoF(biadj_mat, deg_of_freedom=None, method="uniform", variances=None):
    """Assign degrees of freedom (latent variables) of VAE to latent factors from causal structure learning
    Parameters
    ----------
    biadj_mat: biadjacency matrix of MCM
    deg_of_freedom: desired size of latent space of VAE
    method: how to distribute excess degrees of freedom to latent causal factors
    variances: diag of covariance matrix over measurement variables

    Returns
    -------
    redundant_biadj_mat: biadjacency matrix specifing VAE structure from latent space to decoder
    """

    num_cliques, num_obs = biadj_mat.shape
    if deg_of_freedom is None:
        # then default to upper bound; TODO: change to max_intersect_num from medil.ecc_algorithms
        deg_of_freedom = num_obs**2 // 4
    elif deg_of_freedom < num_cliques:
        warnings.warn(
            f"Input `deg_of_freedom={deg_of_freedom}` is less than the {num_cliques} required for the estimated causal structure. `deg_of_freedom` increased to {num_cliques} to compensate."
        )
        deg_of_freedom = num_cliques

    if method == "uniform":
        latents_per_clique = np.ones(num_cliques, int) * (deg_of_freedom // num_cliques)
    elif method == "clique_size":
        latents_per_clique = np.round(
            (biadj_mat.sum(1) / biadj_mat.sum()) * (deg_of_freedom - num_cliques)
        ).astype(int)
    elif method == "tot_var" or method == "avg_var":
        clique_variances = biadj_mat @ variances
        if method == "avg_var":
            clique_variances /= biadj_mat.sum(1)
        clique_variances /= clique_variances.sum()
        latents_per_clique = np.round(
            clique_variances * (deg_of_freedom - num_cliques)
        ).astype(int)

    for _ in range(2):
        remainder = deg_of_freedom - latents_per_clique.sum()
        latents_per_clique[np.argsort(latents_per_clique)[0:remainder]] += 1

    redundant_biadj_mat = np.repeat(biadj_mat, latents_per_clique, axis=0)

    return redundant_biadj_mat


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
