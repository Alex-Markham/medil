Tutorial
========

Quick start
-----------

If you already have a dataset (with each row an observation and each column a feature) loaded into Python, learning a neuro-causal factor analysis model is as simple as:

.. code-block:: python

   >>> from medil import NeuroCausalFactorAnalysis as ncfa
   >>>
   >>> model = ncfa(verbose=True).fit(dataset)


Once training is complete, the automatically created directory ``trained_ncfa/`` contains the learned model saved in the `PyTorch format <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_, along with a ``training.log`` showing losss and time for each epoch, and the `pickled <https://docs.python.org/3/library/pickle.html>`_ training and reconstruction errors.

Or for a linear Gaussian causal factor model:

.. code-block:: python

   >>> from medil import GaussianMCM
   >>>
   >>> model = GaussianMCM().fit(dataset)
   >>> print(model.parameters)


Generate Gaussian MeDIL causal model and sample dataset
-------------------------------------------------------

.. code-block:: python

   >>> from medil import sample
   >>>
   >>> model = sample.mcm(num_meas=5, density=0.3) # generate random MCM graph and parameters
   >>> print(model.parameters)
   parameters.parameterization: Gaussian
   parameters.error_means: [-1.77987111  1.64474766  0.95997585 -0.46474584 -1.60376116]
   parameters.error_variances: [1.16412924 1.89652597 0.56076607 1.59800929 1.42155987]
   parameters.biadj_weights: [[ 0.          1.59352268  0.          1.89113589  0.        ]
                              [-1.95188928  0.         -0.52205946  1.79546014  1.97179256]]
   >>> model.sample(1000) # sample a dataset of 1000 observations from the MCM
   array([[-3.24866352,  2.42247595, -0.22936393, -1.09906127, -2.88687322],
          [-2.84560456,  2.49709866,  0.58608135, -1.05593443, -1.41493064],
          [-2.67268501,  3.71808402, -0.33233638, -0.76753009, -1.30038607],
          ...,
          [-3.41907807,  6.17985143,  0.24999032,  3.22084801, -2.41481477],
          [ 0.61913195,  2.06931685,  0.29016483, -5.46513264, -4.08439981],
          [-1.95685134,  0.95047326,  0.5700815 ,  0.00866948, -1.64228944]])
