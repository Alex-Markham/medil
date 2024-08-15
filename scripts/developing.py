import numpy as np
from medil.sample import mcm, biadj
import matplotlib.pyplot as plt
import seaborn as sns


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

# create arrays to store optimization results
# metric A: squared distance
squared_distance_results_mle = np.zeros((len(lambda_values), len(mu_values)))
squared_distance_results_lse = np.zeros((len(lambda_values), len(mu_values)))

# metric B: sfd
sfd_results_mle = np.zeros((len(lambda_values), len(mu_values)))
sfd_results_lse = np.zeros((len(lambda_values), len(mu_values)))

# Store lambda and mu indices
lambda_indices = {v: i for i, v in enumerate(lambda_values)}
mu_indices = {v: i for i, v in enumerate(mu_values)}

# Initialize counters for failures
mle_failure_count = 0
lse_failure_count = 0

for lambda_reg in lambda_values:
    for mu_reg in mu_values:
        # Get the index
        lambda_idx = lambda_indices[lambda_reg]
        mu_idx = mu_indices[mu_reg]

        # Penalized MLE
        model = DevMedil(biadj=biadj_matrix, rng=rng(), lambda_reg=lambda_reg, mu_reg=mu_reg)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=RuntimeWarning)
                model.fit(dataset, method='mle')

                # Check if any RuntimeWarnings were raised
                if any(issubclass(warning.category, RuntimeWarning) for warning in w):
                    runtime_warning_count_mle += 1

            squared_distance_mle, perm, sfd_value_mle, ushd_value_mle = calculate_metrics(model, 'mle', 0.5, W_star)

            # Store results in arrays
            squared_distance_results_mle[lambda_idx, mu_idx] = squared_distance_mle
            sfd_results_mle[lambda_idx, mu_idx] = sfd_value_mle

            if squared_distance_mle < min_squared_distance_mle:
                min_squared_distance_mle = squared_distance_mle
                best_perm = perm
                best_lambda_mle = lambda_reg
                best_mu_mle = mu_reg

            print(f"lambda_reg={lambda_reg}, mu_reg={mu_reg}, squared_distance_mle={squared_distance_mle}, best_perm={best_perm}")

        except Exception as e:
            mle_failure_count += 1
            print(f"Exception encountered during MLE with lambda_reg={lambda_reg}, mu_reg={mu_reg}: {e}")

        # Penalized LSE
        model = DevMedil(biadj=biadj_matrix, rng=rng(), lambda_reg=lambda_reg, mu_reg=mu_reg)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=RuntimeWarning)
                model.fit(dataset, method='lse')

                # Check if any RuntimeWarnings were raised
                if any(issubclass(warning.category, RuntimeWarning) for warning in w):
                    runtime_warning_count_lse += 1

            squared_distance_lse, perm, sfd_value_lse, ushd_value_lse = calculate_metrics(model, 'lse', 0.5, W_star)

            # Store results in arrays
            squared_distance_results_lse[lambda_idx, mu_idx] = squared_distance_lse
            sfd_results_lse[lambda_idx, mu_idx] = sfd_value_lse

            if squared_distance_lse < min_squared_distance_lse:
                min_squared_distance_lse = squared_distance_lse
                best_perm = perm
                best_lambda_lse = lambda_reg
                best_mu_lse = mu_reg

            print(f"lambda_reg={lambda_reg}, mu_reg={mu_reg}, squared_distance_lse={squared_distance_lse}, best_perm={best_perm}")

        except Exception as e:
            lse_failure_count += 1
            print(f"Exception encountered during LSE with lambda_reg={lambda_reg}, mu_reg={mu_reg}: {e}")