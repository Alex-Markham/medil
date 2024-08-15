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

# define the log scale grid for lambda_reg and mu_reg
lambda_values = np.logspace(-5, 1, num=50)
mu_values = np.logspace(-5, 1, num=50)

# initialize variables to store the best parameters and the minimum squared distance
best_lambda_mle = None
best_mu_mle = None
best_lambda_lse = None
best_mu_lse = None
min_squared_distance_mle = float('inf')
min_squared_distance_lse = float('inf')

# initialize counter for RuntimeWarnings
runtime_warning_count_mle = 0
runtime_warning_count_lse = 0

# real value of w
W_star = generated_sample.parameters.biadj_weights
