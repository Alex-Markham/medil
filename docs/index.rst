Welcome to ``MeDIL``'s documentation!
=====================================

MeDIL is a Python package for causal factor analysis, using the measurement dependence inducing latent (MeDIL) causal model framework :cite:`Markham_2020_UAI`.
The package is under active development---see the ``develop`` branch of the repository on `GitLab <https://gitlab.com/alex-markham/medil/-/tree/develop>`_ or its `Github mirror <https://github.com/Alex-Markham/medil>`_.

Features:
---------

* constraint-based learning of minimum MeDIL causal graphs from marginal independence tests using `distance covariance <https://dcor.readthedocs.io/en/stable/index.html>`_ or `xi correlation <https://pypi.org/project/xicorrelation/>`_

* estimation of causal factor loadings in the linear Gaussian setting or in the nonparametric setting using a variational autoencoder :cite:`markham2023neuro`

* sampling from and random generation of linear Gaussian causal factor models

* implementation of exact algorithm for minimum edge clique cover (ECC) :cite:`Gramm_2009` and wrapper for heuristic minimum ECC written in Java :cite:`conte2020`

Design principles:
------------------

* `scikit-learn <https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects>`_ style API

* basic functionality with minimal dependencies (just `NumPy <https://numpy.org/>`_) and optional dependencies for more functionality

* as much as possible implemented using NumPy ``ndarray``s and methods for fast performance and wide compatibility


Documentation
=============
..
   .. image:: https://gitlab.com/alex-markham/medil/badges/develop/coverage.svg
       :target: https://medil.causal.dev/htmlcov/

:Version: |version|
:Date: |today|

.. toctree::
   :maxdepth: 2

   installation
   theory
   tutorial
   apilist
   license
   changelog
   citing
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
