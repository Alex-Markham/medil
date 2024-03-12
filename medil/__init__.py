"""This package is for causal inference, focusing on the Measurement
Dependence Inducing Latent (MeDIL) Causal Model framework."""
from numpy import median
from .functional_MCM import MedilCausalModel

ncfa = MedilCausalModel(parameterization="vae")
