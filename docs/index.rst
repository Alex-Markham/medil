Welcome to ``MeDIL``'s documentation!
=====================================

MeDIL is a Python package for causal factor analysis, using the measurement dependence inducing latent (MeDIL) causal model framework :cite:`Markham_2020_UAI`.
The package is under active development---see the ``develop`` branch of the repository on `GitLab <https://gitlab.com/alex-markham/medil/-/tree/develop>`_ or its `Github mirror <https://github.com/Alex-Markham/medil>`_.

..
   .. image:: https://gitlab.com/alex-markham/medil/badges/develop/coverage.svg
       :target: https://medil.causal.dev/htmlcov/

:Version: |version|
:Date: |today|

Features:
---------
* estimation of sparse causal factor structure and loadings in the linear Gaussian setting or more generally using a deep generative model :cite:`markham2023neuro`

* :math:`\ell_0`-penalized maximum likelihood estimation (BIC score-based search) for minimum MeDIL causal graphs in the linear Gaussian setting, as well as nonparametric constraint-based search using `distance covariance <https://dcor.readthedocs.io/en/stable/index.html>`_ or `xi correlation <https://pypi.org/project/xicorrelation/>`_

* random generation and sampling from of linear Gaussian causal factor models

* exact search for minimum edge clique cover (ECC) :cite:`Gramm_2009` as well as polynomial time heuristic using the one-pure-child assumption :cite:`markham2023neuro`

Design principles:
------------------
* `scikit-learn <https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects>`_ style API

* basic functionality with minimal dependencies (just `SciPy
  <https://scipy.org>`_) and optional dependencies (`PyTorch
  <https://pytorch.org>`_, `NetworkX <https://networkx.org/>`_, etc.) for more functionality

Further documentation:
----------------------
.. toctree::
   :maxdepth: 2

   installation
   theory
   tutorial
   apilist
   citing
   references
   changelog
   license
