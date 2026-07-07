Theory
======

A *MeDIL causal model* (MCM) :cite:`Markham_2020_UAI` is a bipartite DAG whose vertex set is partitioned into unobserved *latent variables* and observed *measurement variables*, with edges directed only from latents to measurements.
Each measurement has out-degree zero—it is caused by latents but does not cause other variables—and each latent causes at least one measurement.
This *strong causal insufficiency* assumption encodes that all dependences among measurements are due to shared latent causes.
Unlike standard factor analysis, MCMs impose no linearity or distributional constraints; identifiability is obtained instead through causal minimality, as described below.

Under strong causal insufficiency, the only conditional independence (CI) constraints among measurement variables are marginal independence constraints (Proposition 6 of :cite:`Markham_2020_UAI`).
Structure learning therefore reduces to estimating the *undirected dependence graph* (UDG) over measurements—an edge connects :math:`M_i` and :math:`M_j` when they are marginally dependent—via pairwise nonparametric independence tests.
Because many distinct MCMs may be observationally consistent with the same UDG, identifiability is obtained by applying causal minimality: the *minimum MCM* (minMCM) is the observationally consistent model with the fewest latent variables :cite:`markham2023neuro`.

The central result (Proposition 7 of :cite:`Markham_2020_UAI`) is that every minMCM corresponds to a minimum *edge clique cover* (ECC) :cite:`Gramm_2009` of the UDG—a smallest collection of cliques whose union covers every edge—with one latent per clique directed to exactly the clique's measurements.
Two structural properties follow: the latents of any minMCM are pairwise marginally independent (Proposition 9), and a minMCM may require more latents than measurements (Proposition 10), unlike in standard factor analysis.
Finding a minimum ECC is NP-hard; MeDIL provides both an exact branch-and-bound solver :cite:`Gramm_2009` and a polynomial-time heuristic based on the one-pure-child assumption :cite:`markham2023neuro`.

The estimated ``biadj`` can be parameterized in the linear Gaussian setting via :class:`~medil.models.GaussianMCM`, which fits a model in which each measurement is a weighted sum of its latent parents plus independent Gaussian noise, with weights and variances estimated by maximum likelihood.
:class:`~medil.models.NeuroCausalFactorAnalysis` (NCFA; :cite:`markham2023neuro`) instead learns a variational autoencoder (VAE) whose decoder is constrained to the structure of ``biadj``, so each measurement's generative network receives inputs only from its assigned latent causes, and parameters are estimated by maximizing the evidence lower bound (ELBO).
