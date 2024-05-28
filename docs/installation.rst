
Installation
============

You can install the package from `PyPI <https://pypi.org/project/medil/>`_ with the command ``pip install medil``.
The default only requires SciPy and allows for learning causal factor models from data, random generation of causal factor models, and sampling data from a given (or learned) causal factor model, all in the linear Gaussian setting.
There are additional optional requirements for extra features:

+-------+--------------------------------+
|Key    |Description                     |
+-------+--------------------------------+
|``all``|Installs all                    |
|       |optional                        |
|       |dependencies,                   |
|       |for full                        |
|       |functionality;                  |
|       |equivalent to                   |
|       |using all of                    |
|       |the other keys                  |
|       |below.                          |
+-------+--------------------------------+
|``dgm``|Uses deep generative models for |
|       |the causal mechanisms (instead  |
|       |of restricting to the linear    |
|       |Gaussian setting), specifically |
|       |using `PyTorch                  |
|       |<https://pytorch.org/docs/>`_ to|
|       |implement a variational         |
|       |autoencoder.                    |
+-------+--------------------------------+
|``vis``|Uses `NetworkX                  |
|       |<https://networkx.github.io/>`_,|
|       |`Matplotlib                     |
|       |<https://matplotlib.org/>`_ and |
|       |`seaborn                        |
|       |<https://seaborn.pydata.org/>`_ |
|       |for vizualize the causal factor |
|       |graph and various plots.        |
+-------+--------------------------------+

In order to use these features, simply append the corresponding comma-separated keys after the package name in brackets (with no spaces), e.g., ``pip install medil[vis]``.
