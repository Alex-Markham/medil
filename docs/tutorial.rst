Tutorial
========

Quick start
-----------

If you already have a dataset (with each row an observation and each column a feature) loaded into Python, learning a neuro-causal factor analysis model is as simple as:

.. code-block:: python

   from medil import NeuroCausalFactorAnalysis as ncfa

   model = ncfa(verbose=True).fit(dataset)


Once training is complete, the automatically created directory ``trained_ncfa/`` contains the learned model saved in the `PyTorch format <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_, along with a ``training.log`` showing losss and time for each epoch, and the `pickled <https://docs.python.org/3/library/pickle.html>`_ training and reconstruction errors.

Or for a linear Gaussian causal factor model:

.. code-block:: python

   from medil import GaussianMCM

   model = GaussianMCM().fit(dataset)

   print(model.parameters)
