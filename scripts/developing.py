import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from medil.models import DevMedil
from medil.evaluate import sfd, min_perm_squared_l2_dist
from medil.sample import mcm, biadj


# Set the random variable generator seed
def rng():
    return np.random.default_rng(3)


# define a function for model fitting and metric calculation
def calculate_metrics(model, method, threshold, W_star):
    # mle and lse
    if method == "mle":
        W_hat = model.W_hat_mle
    else:
        W_hat = model.W_hat_lse
    # metric A
    perm, squared_distance = min_perm_squared_l2_dist(W_hat, W_star)

    # metric B
    W_hat_zero_pattern = (np.abs(W_hat) > threshold).astype(int)
    sfd_value, ushd_value = sfd(biadj_matrix, W_hat_zero_pattern)

    return squared_distance, perm, sfd_value, ushd_value


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
lambda_values = np.logspace(-5, 1, num=20)
mu_values = np.logspace(-5, 1, num=20)

# initialize variables to store the best parameters and the minimum squared distance
best_lambda_mle = None
best_mu_mle = None
best_lambda_lse = None
best_mu_lse = None
min_squared_distance_mle = float("inf")
min_squared_distance_lse = float("inf")

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
        model = DevMedil(
            biadj=biadj_matrix, rng=rng(), lambda_reg=lambda_reg, mu_reg=mu_reg
        )
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=RuntimeWarning)
                model.fit(dataset, method="mle")

                # Check if any RuntimeWarnings were raised
                if any(issubclass(warning.category, RuntimeWarning) for warning in w):
                    runtime_warning_count_mle += 1

            squared_distance_mle, perm, sfd_value_mle, ushd_value_mle = (
                calculate_metrics(model, "mle", 0.5, W_star)
            )

            # Store results in arrays
            squared_distance_results_mle[lambda_idx, mu_idx] = squared_distance_mle
            sfd_results_mle[lambda_idx, mu_idx] = sfd_value_mle

            if squared_distance_mle < min_squared_distance_mle:
                min_squared_distance_mle = squared_distance_mle
                best_perm = perm
                best_lambda_mle = lambda_reg
                best_mu_mle = mu_reg

            print(
                f"lambda_reg={lambda_reg}, mu_reg={mu_reg}, squared_distance_mle={squared_distance_mle}, best_perm={best_perm}"
            )

        except Exception as e:
            mle_failure_count += 1
            print(
                f"Exception encountered during MLE with lambda_reg={lambda_reg}, mu_reg={mu_reg}: {e}"
            )

        # Penalized LSE
        model = DevMedil(
            biadj=biadj_matrix, rng=rng(), lambda_reg=lambda_reg, mu_reg=mu_reg
        )
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=RuntimeWarning)
                model.fit(dataset, method="lse")

                # Check if any RuntimeWarnings were raised
                if any(issubclass(warning.category, RuntimeWarning) for warning in w):
                    runtime_warning_count_lse += 1

            squared_distance_lse, perm, sfd_value_lse, ushd_value_lse = (
                calculate_metrics(model, "lse", 0.5, W_star)
            )

            # Store results in arrays
            squared_distance_results_lse[lambda_idx, mu_idx] = squared_distance_lse
            sfd_results_lse[lambda_idx, mu_idx] = sfd_value_lse

            if squared_distance_lse < min_squared_distance_lse:
                min_squared_distance_lse = squared_distance_lse
                best_perm = perm
                best_lambda_lse = lambda_reg
                best_mu_lse = mu_reg

            print(
                f"lambda_reg={lambda_reg}, mu_reg={mu_reg}, squared_distance_lse={squared_distance_lse}, best_perm={best_perm}"
            )

        except Exception as e:
            lse_failure_count += 1
            print(
                f"Exception encountered during LSE with lambda_reg={lambda_reg}, mu_reg={mu_reg}: {e}"
            )

print(f"Total MLE optimization failures: {mle_failure_count}")
print(f"Total LSE optimization failures: {lse_failure_count}")
print(f"Total RuntimeWarnings encountered during MLE: {runtime_warning_count_mle}")
print(f"Total RuntimeWarnings encountered during LSE: {runtime_warning_count_lse}")

# Plot heatmap for Squared Distance (MLE)
plt.figure(figsize=(10, 8))
sns.heatmap(
    squared_distance_results_mle,
    xticklabels=[f"{x:.2g}" for x in mu_values],
    yticklabels=[f"{y:.2g}" for y in lambda_values],
    cmap="Reds",
)
plt.title("Squared Distance for Penalized MLE")
plt.xlabel("mu_reg")
plt.ylabel("lambda_reg")
plt.savefig("dist_mle.png")

# Plot heatmap for SFD Value (MLE)
plt.figure(figsize=(10, 8))
sns.heatmap(
    sfd_results_mle,
    xticklabels=[f"{x:.2g}" for x in mu_values],
    yticklabels=[f"{y:.2g}" for y in lambda_values],
    cmap="Blues",
)
plt.title("SFD Value for Penalized MLE")
plt.xlabel("mu_reg")
plt.ylabel("lambda_reg")
plt.savefig("sfd_mle.png")

# Plot heatmap for Squared Distance (LSE)
plt.figure(figsize=(10, 8))
sns.heatmap(
    squared_distance_results_lse,
    xticklabels=[f"{x:.2g}" for x in mu_values],
    yticklabels=[f"{y:.2g}" for y in lambda_values],
    cmap="Reds",
)
plt.title("Squared Distance for Penalized LSE")
plt.xlabel("mu_reg")
plt.ylabel("lambda_reg")
plt.savefig("dist_lse.png")

# Plot heatmap for SFD Value (LSE)
plt.figure(figsize=(10, 8))
sns.heatmap(
    sfd_results_lse,
    xticklabels=[f"{x:.2g}" for x in mu_values],
    yticklabels=[f"{y:.2g}" for y in lambda_values],
    cmap="Blues",
)
plt.title("SFD Value for Penalized LSE")
plt.xlabel("mu_reg")
plt.ylabel("lambda_reg")
plt.savefig("sfd_lse.png")

print(f"Best lambda_reg for MLE: {best_lambda_mle}")
print(f"Best mu_reg for MLE: {best_mu_mle}")
print(f"Minimum squared distance (MLE): {min_squared_distance_mle}\n")

print(f"Best lambda_reg for LSE: {best_lambda_lse}")
print(f"Best mu_reg for LSE: {best_mu_lse}")
print(f"Minimum squared distance (LSE): {min_squared_distance_lse}")

## Examine param values from best run:
# best penalized LSE
model = DevMedil(
    biadj=biadj_matrix, rng=rng(), lambda_reg=best_lambda_lse, mu_reg=best_mu_lse
)
model.fit(dataset, method="lse")
W_hat_lse = model.W_hat_lse
D_hat_lse = model.D_hat_lse

model = DevMedil(
    biadj=biadj_matrix, rng=rng(), lambda_reg=best_lambda_mle, mu_reg=best_mu_mle
)
model.fit(dataset, method="mle")
W_hat_mle = model.W_hat_mle
D_hat_mle = model.D_hat_mle

# Set a threshold
threshold = 0.5

# the w_hat and the zero patterns by lse
W_hat_lse_zero_pattern = (np.abs(W_hat_lse) > threshold).astype(int)
print("W_hat_lse Zero Pattern:")
print(W_hat_lse_zero_pattern)

# Compute the squared distance as the evaluation metric
lse_order, squared_distance_lse = min_perm_squared_l2_dist(W_hat_lse, W_star)
print("Squared distance between W_hat_lse and W_star:\n", squared_distance_lse)
mle_order, squared_distance_mle = min_perm_squared_l2_dist(W_hat_mle, W_star)
print("Squared distance between W_hat_mle and W_star:\n", squared_distance_mle)

W_star = generated_sample.parameters.biadj_weights
print("True weights matrix W_star:\n", W_star)
print("Estimated W (LSE):\n", W_hat_lse[np.array(lse_order)])
print("Estimated W (MLE):\n", W_hat_mle[np.array(mle_order)])


D_star = generated_sample.parameters.error_variances
print("\nTrue variances D_star:\n", D_star)
print("D_hat_lse:\n", D_hat_lse)
print("D_hat_mle:\n", D_hat_mle)
