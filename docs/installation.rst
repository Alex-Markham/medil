
Installation
============

You can install the package from `PyPI <https://pypi.org/project/medil/>`_ with the command ``pip install medil``.

The default install covers the linear Gaussian setting, with only NumPy, SciPy, and scikit-learn as dependencies.
To also use :class:`~medil.models.NeuroCausalFactorAnalysis`, install the ``ncfa`` extra, which adds PyTorch::

   pip install medil[ncfa]
