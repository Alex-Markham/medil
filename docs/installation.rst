Installation
============

You can install the package using pip by running ``pip install medi`` in a terminal.
The default only requires NumPy and only allows for limited data simulation, linear Pearson correlation independence testing, and structure learning.
There are additional optional requirements for extra features:

+------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Key        | Description                                                                                                                                                                     | 
+============+=================================================================================================================================================================================+
| ``'dcor'`` | Uses `dcor <https://dcor.readthedocs.io/>`_ to compute the (nonlinear) distance correlation.                                                                                    |
+------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``'gan'``  | Uses `PyTorch <https://pytorch.org/>`_ to make generative adversarial networks (GANs) for advanced data simulation and learning the functional MeDIL causal model.              |
+------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``'vis'``  | Uses `NetworkX <https://networkx.github.io/>`_, `matplotlib <https://matplotlib.org/>`_ and `seaborn <https://seaborn.pydata.org/>`_ for graph visualization and various plots. |
+------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

In order to use these features, simply append the corresponding keys after the package name, e.g., ``pip install medil['vis', 'dcor']``.

If you have problems installing ``dcor``, try using a fresh python 3.7 virtual environment, and first installing ``numba``, and then ``scipy`` before installing ``medil`` with the ``'dcor'`` key.
