MeDIL Causal Modelling Framework
================================

:cite:`Markham_2020_UAI` introduce measurement dependence inducing latent (MeDIL) causal models.
These models have disjoint sets of (unobserved) latent variables and (observed) measurement variables.
In order for a set of random variables to be considered measurement variables, it must satisfy the assumption of strong causal insufficiency, i.e., none of the measurement variables may (even indirectly) cause one another—thus, any probabilistic dependence between them must be mediated by latent causes.
The assumption of strong causal insufficiency is especially applicable in settings such as psychometric instrument questionnaires, and MeDIL causal models can, for example, be thought of as a causally interpretable factor analysis.

For more details, supporting theory, and related work, see :cite:`Markham_2020_UAI` focusing on the graphical structure and :cite:`markham2023neuro` focusing on parameter estimation and deep generative models for the causal mechanisms.
