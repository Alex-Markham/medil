Tutorial
========

Quick start
-----------

The simplest way to get started is to generate a synthetic dataset and fit a model to it:

.. code-block:: python

   >>> from medil import GaussianMCM, NeuroCausalFactorAnalysis, sample
   >>>
   >>> # Generate a random causal model and draw 1000 observations from it
   >>> true_model = sample.mcm(num_meas=6, density=0.3)
   >>> dataset = true_model.sample(1000)
   >>>
   >>> # Fit a linear Gaussian model (structure + parameters learned jointly)
   >>> model = GaussianMCM().fit(dataset)
   >>> print(model.biadj)   # learned bipartite graph
   >>> print(model.parameters)

For the nonlinear setting, standardize first and use ``NeuroCausalFactorAnalysis``:

.. code-block:: python

   >>> dataset = (dataset - dataset.mean(0)) / dataset.std(0)
   >>> model = NeuroCausalFactorAnalysis(verbose=True).fit(dataset)

This first learns a minimal causal factor structure (``model.biadj``) and then a deep generative model from latents to measurements (a masked VAE stored in ``model.parameters.vae``).

To save training artifacts, pass a ``log_path`` argument; MeDIL will create
that directory and write the learned model (in
`PyTorch format <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_)
and pickled training/reconstruction losses to it.


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

You can also generate a randomly initialized NCFA model (useful for simulating
from a nonlinear model before fitting):

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

   >>> from medil.evaluate import sfd
   >>>
   >>> true_biadj = true_model.biadj
   >>> learned_biadj = model.biadj
   >>>
   >>> sfd(true_biadj, learned_biadj)                    # structural Frobenius distance (int)
   >>> sfd(true_biadj, learned_biadj, to_return="both")  # (raw, normalized)

Lower is better. SFD compares the weighted undirected graphs induced by each biadjacency matrix.


Tuning NeuroCausalFactorAnalysis
---------------------------------

Training hyperparameters are exposed via the ``hyperparams`` dict and can be
changed before calling :meth:`fit`:

.. code-block:: python

   >>> model = NeuroCausalFactorAnalysis()
   >>> model.hyperparams.update({
   ...     "num_epochs": 500,
   ...     "lr": 5e-4,
   ...     "beta": 0.5,          # down-weight KL term
   ...     "latent_width": 4,    # wider latent representations
   ...     "early_stopping": True,
   ...     "patience": 30,
   ... })
   >>> model.fit(dataset)

See the :class:`~medil.models.NeuroCausalFactorAnalysis` API docs for the
full list of keys and their defaults.


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
