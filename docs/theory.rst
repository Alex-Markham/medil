MeDIL Causal Modelling Framework
================================

:cite:`Markham_2020_UAI` introduce measurement dependence inducing latent (MeDIL) causal models.
These models have disjoint sets of (unobserved) latent variables and (observed) measurement variables.
In order for a set of random variables to be considered measurement variables, it must satisfy the assumption of strong causal insufficiency, i.e., none of the measurement variables may (even indirectly) cause one another—thus, any probabilistic dependence between them must be mediated by latent causes.
The assumption of strong causal insufficiency is especially applicable in settings such as psychometric instrument questionnaires, and MeDIL causal models can, for example, be thought of as a causally interpretable factor analysis.

Graphically, MeDIL causal models (MCMs) are represented as directed acyclic graphs with disjoint sets of vertices representing the latent and measurement variables, where the measurement variables are represented as sink vertices (i.e., have no outgoing edges).
These MCMs can be inferred by sampling a set of measurement variables as follows:

1. perform (nonlinear) independence tests on samples to generate undirected dependency graph (UDG) over measurement variables
2. perform causal structure learning by applying an edge clique cover finding algorithm to the UDG, resulting in a graphical MCM
3. use generative adversarial networks to learn a functional MCM (i.e., learn the functional relations corresponding to edges in the to the graphical MCM)

See :cite:`Markham_2020_UAI` for more details, supporting theory, and related work for steps 1 and 2, and see :cite:`Chivukula_2020` for those of step 3.
