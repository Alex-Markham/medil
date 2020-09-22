
Installation
============

You can install the package using pip by running ``pip install medil`` in a terminal.
The default only requires NumPy and only allows for limited data simulation, linear Pearson correlation independence testing, and structure learning.
There are additional optional requirements for extra features:

+----------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Key      | Description                                                                                                                                                                                      | 
+==========+==================================================================================================================================================================================================+
| ``all``  | Installs all optional dependencies, for full functionality; equivalent to using all of the other keys below                                                                                      |
+----------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``dcor`` | Uses `dcor <https://dcor.readthedocs.io/>`_ to compute the (nonlinear) distance correlation.                                                                                                     |
+----------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``gan``  | Uses `PyTorch Lightning <https://pytorch-lightning.readthedocs.io>`_ to make generative adversarial networks (GANs) for advanced data simulation and learning the functional MeDIL causal model. |
+----------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``vis``  | Uses `NetworkX <https://networkx.github.io/>`_, `matplotlib <https://matplotlib.org/>`_ and `seaborn <https://seaborn.pydata.org/>`_ for graph visualization and various plots.                  |
+----------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

In order to use these features, simply append the corresponding comma-separated keys after the package name in brackets (with no spaces), e.g., ``pip install medil[vis,dcor]``.

You may encounter problems with installation due to ``llvmlite`` which is required by ``numba`` which is required by ``dcor``.
These can usually be resolved by using a fresh virtual environment and then installing ``medil`` (with the ``all`` or ``dcor`` key) before any other packages.
