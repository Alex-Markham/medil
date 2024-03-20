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
