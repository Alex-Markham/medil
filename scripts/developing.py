import numpy as np
from medil.sample import mcm, biadj


# Set the random variable generator seed
def rng():
    return np.random.default_rng(3)


# Set up the parameters
num_meas = 3
density = 0.6
num_latent = 2
# Generate the bipartite adjacency matrix
biadj_matrix = biadj(
    num_meas=num_meas, density=density, num_latent=num_latent, rng=rng()
)

# Generate the MCM sample
generated_sample = mcm(rng=rng(), parameterization="Gaussian", biadj=biadj_matrix)

# Generate the dataset
dataset = generated_sample.sample(1000)
