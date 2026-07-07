Tutorial
========

Quick start
-----------

If you already have a dataset (with each row an observation and each column a feature) loaded into Python, learning a neuro-causal factor analysis model is as simple as:

.. code-block:: python

   >>> from medil import NeuroCausalFactorAnalysis
   >>>
   >>> model = NeuroCausalFactorAnalysis(verbose=True).fit(dataset)

This jointly learns the causal factor structure (``model.biadj``) and the nonlinear generative mechanisms (a masked VAE stored in ``model.parameters.vae``).

To save training artifacts, pass a ``log_path`` argument; MeDIL will create that directory and write the learned model (in `PyTorch format <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_) and pickled training/reconstruction errors to it.

For a linear Gaussian causal factor model:

.. code-block:: python

   >>> from medil import GaussianMCM
   >>>
   >>> model = GaussianMCM().fit(dataset)
   >>> print(model.parameters)


Structure learning only
-----------------------

If you only need the undirected dependence graph (UDG) over observed variables, use ``estimate_UDG`` directly:

.. code-block:: python

   >>> from medil.independence_testing import estimate_UDG
   >>>
   >>> udg, p_values = estimate_UDG(dataset, method="dcov_fast")

Available methods are ``"dcov_fast"`` (default), ``"dcov"`` (slower but more accurate for small samples), and ``"xicor"``.
The returned ``udg`` is a boolean adjacency matrix over the measurement variables; ``p_values`` is the corresponding matrix of test p-values.


Sampling
--------

Generate a random Gaussian MeDIL causal model and draw a synthetic dataset:

.. code-block:: python

   >>> from medil import sample
   >>>
   >>> model = sample.mcm(num_meas=5, density=0.3)
   >>> print(model.parameters)
   parameters.parameterization: Gaussian
   parameters.error_means: [-1.77987111  1.64474766  0.95997585 -0.46474584 -1.60376116]
   parameters.error_variances: [1.16412924 1.89652597 0.56076607 1.59800929 1.42155987]
   parameters.biadj_weights: [[ 0.          1.59352268  0.          1.89113589  0.        ]
                              [-1.95188928  0.         -0.52205946  1.79546014  1.97179256]]
   >>> dataset = model.sample(1000)

You can also generate a randomly initialized NCFA model (useful for simulating from a nonlinear model before fitting):

.. code-block:: python

   >>> ncfa_model = sample.mcm(num_meas=5, density=0.3, parameterization="VAE")
   >>> dataset = ncfa_model.sample(1000)

Once an NCFA model is fitted, sampling works the same way:

.. code-block:: python

   >>> fitted = NeuroCausalFactorAnalysis().fit(dataset)
   >>> new_samples = fitted.sample(500)                          # shape (500, num_meas)
   >>> new_samples, latents = fitted.sample(500, include_latent=True)  # also return latent codes


Evaluation
----------

Given a known ground-truth structure (e.g. from a simulation), measure how close the learned graph is:

.. code-block:: python

   >>> from medil.evaluate import sfd, shd
   >>>
   >>> true_biadj = model.biadj        # e.g. from sample.mcm(...)
   >>> learned_biadj = fitted.biadj
   >>>
   >>> sfd(true_biadj, learned_biadj)                    # structural Frobenius distance (int)
   >>> sfd(true_biadj, learned_biadj, to_return="both")  # (raw, normalized)
   >>>
   >>> shd(true_biadj, predicted_biadj=learned_biadj)                    # structural Hamming distance
   >>> shd(true_biadj, predicted_biadj=learned_biadj, to_return="both")  # (raw, normalized)

Lower is better for both metrics. SFD compares the weighted undirected graphs induced by each biadj; SHD counts incorrect edges in the recovered undirected dependence graph.


Accessing model internals
-------------------------

For a fitted ``GaussianMCM``:

.. code-block:: python

   >>> model = GaussianMCM().fit(dataset)
   >>> model.biadj                      # boolean (num_latent, num_meas) biadjacency matrix
   >>> model.parameters.biadj_weights   # float edge weights, zero where biadj is False
   >>> model.parameters.error_means     # per-measurement noise means
   >>> model.parameters.error_variances # per-measurement noise variances

For a fitted ``NeuroCausalFactorAnalysis``:

.. code-block:: python

   >>> model = NeuroCausalFactorAnalysis().fit(dataset)
   >>> model.biadj               # learned boolean (num_latent, num_meas) structure
   >>> model.parameters.vae      # the trained VariationalAutoencoder (PyTorch module)
   >>> model.loss                # dict with train/valid ELBO and reconstruction losses
