import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from medil.models import DevMedil, GaussianMCM
from medil.evaluate import sfd, min_perm_squared_l2_dist_abs
from medil.sample import mcm, biadj

from sklearn.model_selection import KFold


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
    perm, squared_distance = min_perm_squared_l2_dist_abs(W_hat, W_star)

    # metric B
    W_hat_zero_pattern = (np.abs(W_hat) > threshold).astype(int)
    true_zero_pattern = W_star.astype(bool)
    sfd_value, ushd_value = sfd(W_hat_zero_pattern, true_zero_pattern)

    return squared_distance, perm, sfd_value, ushd_value


# calculate validation error using sample covariance matrix
def calculate_validation_error(model, method, held_out_data, lambda_reg, mu_reg):
    if method == "lse":
        return model.validation_lse(lambda_reg, mu_reg, held_out_data)
    elif method == "mle":
        return model.validation_mle(lambda_reg, mu_reg, held_out_data)


# Hyperparameter tuning
def grid_search(true_model, dataset, verbose=False):
    # define the log scale grid for lambda_reg and mu_reg
    lambda_values = np.logspace(-6, 1, num=6)
    mu_values = np.logspace(-6, 1, num=6)

    # initialize variables to store the best parameters and the minimum squared distance
    best_lambda_lse = None
    best_mu_lse = None
    best_perm_lse = None
    best_lambda_mle = None
    best_mu_mle = None
    best_perm_mle = None
    min_squared_distance_mle = float("inf")
    min_squared_distance_lse = float("inf")

    # initialize counter for RuntimeWarnings
    runtime_warning_count_mle = 0
    runtime_warning_count_lse = 0

    # real value of w
    W_star = true_model.parameters.biadj_weights

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
            model = DevMedil(rng=rng())
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", category=RuntimeWarning)
                    model.fit(
                        dataset, method="mle", lambda_reg=lambda_reg, mu_reg=mu_reg
                    )

                    # Check if any RuntimeWarnings were raised
                    if any(
                        issubclass(warning.category, RuntimeWarning) for warning in w
                    ):
                        runtime_warning_count_mle += 1

                squared_distance_mle, perm, sfd_value_mle, ushd_value_mle = (
                    calculate_metrics(model, "mle", 0.5, W_star)
                )

                # Store results in arrays
                squared_distance_results_mle[lambda_idx, mu_idx] = squared_distance_mle
                sfd_results_mle[lambda_idx, mu_idx] = sfd_value_mle

                if squared_distance_mle < min_squared_distance_mle:
                    min_squared_distance_mle = squared_distance_mle
                    best_perm_mle = perm
                    best_lambda_mle = lambda_reg
                    best_mu_mle = mu_reg

                if verbose:
                    print(
                        f"lambda_reg={lambda_reg}, mu_reg={mu_reg}, squared_distance_mle={squared_distance_mle}, best_perm={best_perm_mle}"
                    )

            except Exception as e:
                mle_failure_count += 1
                if verbose:
                    print(
                        f"Exception encountered during MLE with lambda_reg={lambda_reg}, mu_reg={mu_reg}: {e}"
                    )

            # Penalized LSE
            model = DevMedil(rng=rng())
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", category=RuntimeWarning)
                    model.fit(
                        dataset, method="lse", lambda_reg=lambda_reg, mu_reg=mu_reg
                    )

                    # Check if any RuntimeWarnings were raised
                    if any(
                        issubclass(warning.category, RuntimeWarning) for warning in w
                    ):
                        runtime_warning_count_lse += 1

                squared_distance_lse, perm, sfd_value_lse, ushd_value_lse = (
                    calculate_metrics(model, "lse", 0.5, W_star)
                )

                # Store results in arrays
                squared_distance_results_lse[lambda_idx, mu_idx] = squared_distance_lse
                sfd_results_lse[lambda_idx, mu_idx] = sfd_value_lse

                if squared_distance_lse < min_squared_distance_lse:
                    min_squared_distance_lse = squared_distance_lse
                    best_perm_lse = perm
                    best_lambda_lse = lambda_reg
                    best_mu_lse = mu_reg

                if verbose:
                    print(
                        f"lambda_reg={lambda_reg}, mu_reg={mu_reg}, squared_distance_lse={squared_distance_lse}, best_perm={best_perm_lse}"
                    )

            except Exception as e:
                lse_failure_count += 1
                if verbose:
                    print(
                        f"Exception encountered during LSE with lambda_reg={lambda_reg}, mu_reg={mu_reg}: {e}"
                    )
    if verbose:
        print(f"Total MLE optimization failures: {mle_failure_count}")
        print(f"Total LSE optimization failures: {lse_failure_count}")
        print(
            f"Total RuntimeWarnings encountered during MLE: {runtime_warning_count_mle}"
        )
        print(
            f"Total RuntimeWarnings encountered during LSE: {runtime_warning_count_lse}"
        )

    return (
        W_star,
        squared_distance_results_lse,
        squared_distance_results_mle,
        sfd_results_lse,
        sfd_results_mle,
        mu_values,
        lambda_values,
        best_mu_lse,
        best_mu_mle,
        best_lambda_mle,
        best_lambda_lse,
        min_squared_distance_lse,
        min_squared_distance_mle,
    )


# Hyperparameter tuning with K-fold cross-validation
def grid_search_kfold(true_model, dataset, k=5, verbose=False):
    lambda_values = np.logspace(-6, 1, num=6)
    mu_values = np.logspace(-6, 1, num=6)
    W_star = true_model.parameters.biadj_weights
    kf = KFold(n_splits=k, shuffle=True, random_state=3)

    validation_error_results_mle = np.zeros((len(lambda_values), len(mu_values)))
    validation_error_results_lse = np.zeros((len(lambda_values), len(mu_values)))
    squared_distance_results_mle = np.zeros((len(lambda_values), len(mu_values)))
    squared_distance_results_lse = np.zeros((len(lambda_values), len(mu_values)))
    sfd_results_mle = np.zeros((len(lambda_values), len(mu_values)))
    sfd_results_lse = np.zeros((len(lambda_values), len(mu_values)))

    lambda_indices = {v: i for i, v in enumerate(lambda_values)}
    mu_indices = {v: i for i, v in enumerate(mu_values)}

    for lambda_reg in lambda_values:
        for mu_reg in mu_values:
            fold_validation_errors_mle = []
            fold_validation_errors_lse = []
            fold_true_errors_mle = []
            fold_true_errors_lse = []
            fold_sfd_value_mle = []
            fold_sfd_value_lse = []

            for train_index, val_index in kf.split(dataset):
                train_data = dataset[train_index]
                val_data = dataset[val_index]

                # Penalized MLE
                model = DevMedil(rng=rng())
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", category=RuntimeWarning)
                        model.fit(
                            train_data,
                            method="mle",
                            lambda_reg=lambda_reg,
                            mu_reg=mu_reg,
                        )

                    val_error_mle = calculate_validation_error(
                        model, "mle", val_data, lambda_reg, mu_reg
                    )
                    fold_validation_errors_mle.append(val_error_mle)

                    true_error_mle, _, sfd_value_mle, _ = calculate_metrics(
                        model, "mle", 0.5, W_star
                    )
                    fold_true_errors_mle.append(true_error_mle)
                    fold_sfd_value_mle.append(sfd_value_mle)

                except Exception as e:
                    if verbose:
                        print(
                            f"Exception during MLE with lambda_reg={lambda_reg}, mu_reg={mu_reg}: {e}"
                        )

                # Penalized LSE
                model = DevMedil(rng=rng())
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", category=RuntimeWarning)
                        model.fit(
                            train_data,
                            method="lse",
                            lambda_reg=lambda_reg,
                            mu_reg=mu_reg,
                        )

                    val_error_lse = calculate_validation_error(
                        model, "lse", val_data, lambda_reg, mu_reg
                    )
                    fold_validation_errors_lse.append(val_error_lse)

                    true_error_lse, _, sfd_value_lse, _ = calculate_metrics(
                        model, "lse", 0.5, W_star
                    )
                    fold_true_errors_lse.append(true_error_lse)
                    fold_sfd_value_lse.append(sfd_value_lse)

                except Exception as e:
                    if verbose:
                        print(
                            f"Exception during LSE with lambda_reg={lambda_reg}, mu_reg={mu_reg}: {e}"
                        )

            avg_val_error_mle = np.mean(fold_validation_errors_mle)
            avg_true_error_mle = np.mean(fold_true_errors_mle)
            avg_sfd_value_mle = np.mean(fold_sfd_value_mle)

            avg_val_error_lse = np.mean(fold_validation_errors_lse)
            avg_true_error_lse = np.mean(fold_true_errors_lse)
            avg_sfd_value_lse = np.mean(fold_sfd_value_lse)

            validation_error_results_mle[
                lambda_indices[lambda_reg], mu_indices[mu_reg]
            ] = avg_val_error_mle
            validation_error_results_lse[
                lambda_indices[lambda_reg], mu_indices[mu_reg]
            ] = avg_val_error_lse
            squared_distance_results_mle[
                lambda_indices[lambda_reg], mu_indices[mu_reg]
            ] = avg_true_error_mle
            squared_distance_results_lse[
                lambda_indices[lambda_reg], mu_indices[mu_reg]
            ] = avg_true_error_lse
            sfd_results_mle[lambda_indices[lambda_reg], mu_indices[mu_reg]] = (
                avg_sfd_value_mle
            )
            sfd_results_lse[lambda_indices[lambda_reg], mu_indices[mu_reg]] = (
                avg_sfd_value_lse
            )

            if verbose:
                print(
                    f"lambda_reg={lambda_reg}, mu_reg={mu_reg}, avg_val_error_mle={avg_val_error_mle}, avg_true_error_mle={avg_true_error_mle}"
                )
                print(
                    f"lambda_reg={lambda_reg}, mu_reg={mu_reg}, avg_val_error_lse={avg_val_error_lse}, avg_true_error_lse={avg_true_error_lse}"
                )

    return (
        validation_error_results_lse,
        validation_error_results_mle,
        squared_distance_results_lse,
        squared_distance_results_mle,
        sfd_results_lse,
        sfd_results_mle,
        mu_values,
        lambda_values,
    )


# define fixed_biadj_mat_list
fixed_biadj_mat_list = [
    np.array([[True, True]]),
    np.array([[True, True, True]]),
    np.array([[True, True, True, True]]),
    np.array([[True, True, False], [False, True, True]]),
    np.array([[True, True, True, False, False], [False, False, True, True, True]]),
    np.array(
        [
            [True, True, True, False, False, False, False],
            [False, False, True, True, True, False, False],
            [False, False, False, False, True, True, True],
        ]
    ),
    # np.array(
    #     [
    #         [True, True, True, False, False, False, False, False, False],
    #         [False, False, True, True, True, False, False, False, False],
    #         [False, False, False, False, True, True, True, False, False],
    #         [False, False, False, False, False, False, True, True, True],
    #     ]
    # ),
    # np.array(
    #     [
    #         [True, True, True, False, False, False, False, False, False, False, False],
    #         [False, False, True, True, True, False, False, False, False, False, False],
    #         [False, False, False, False, True, True, True, False, False, False, False],
    #         [False, False, False, False, False, False, True, True, True, False, False],
    #         [False, False, False, False, False, False, False, False, True, True, True],
    #     ]
    # ),
    np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ],
        dtype=bool,
    ),
]


# examine weight matrix recovery, regularizers, and optimization
# results for benchmark graphs (toy model is the 4th graph)
def benchmark_graphs_deep_dive(fixed_biadj_mat_list, verbose=False):
    for idx, biadj_matrix in enumerate(fixed_biadj_mat_list):
        print(f"\nTesting Graph {idx + 1} with shape {biadj_matrix.shape}")
        num_meas, num_latent = biadj_matrix.shape

        # Generate the MCM sample
        true_model = mcm(rng=rng(), parameterization="Gaussian", biadj=biadj_matrix)

        # Generate the dataset
        dataset = true_model.sample(5000)

        # Run grid search
        (
            W_star,
            squared_distance_results_lse,
            squared_distance_results_mle,
            sfd_results_lse,
            sfd_results_mle,
            mu_values,
            lambda_values,
            best_mu_lse,
            best_mu_mle,
            best_lambda_mle,
            best_lambda_lse,
            min_squared_distance_lse,
            min_squared_distance_mle,
        ) = grid_search(true_model, dataset, verbose=verbose)

        # Fit the GaussianMCM model
        model = GaussianMCM(biadj=biadj_matrix, rng=rng())
        model.fit(dataset)

        # Calculate W_hat from the GaussianMCM model
        W_hat_gaussian = model.parameters.biadj_weights

        # Calculate the squared distance between W_hat_gaussian and W_star
        _, squared_distance_gaussian = min_perm_squared_l2_dist_abs(
            W_hat_gaussian, W_star
        )

        print(
            f"Squared distance between W_hat_gaussian and W_star (Graph {idx + 1}):\n {squared_distance_gaussian}"
        )
        print(f"True weights:\n {true_model.parameters.biadj_weights}")
        print(f"GaussianMCM learned weights:\n {W_hat_gaussian}")

        # # Plot heatmap for Squared Distance (LSE)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(
        #     squared_distance_results_lse,
        #     xticklabels=[f"{x:.2g}" for x in mu_values],
        #     yticklabels=[f"{y:.2g}" for y in lambda_values],
        #     cmap="Reds",
        # )
        # plt.title("Squared Distance for Penalized LSE")
        # plt.xlabel("mu_reg")
        # plt.ylabel("lambda_reg")
        # plt.savefig("lse_dist.png")

        # # Plot heatmap for SFD Value (LSE)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(
        #     sfd_results_lse,
        #     xticklabels=[f"{x:.2g}" for x in mu_values],
        #     yticklabels=[f"{y:.2g}" for y in lambda_values],
        #     cmap="Blues",
        # )
        # plt.title("SFD Value for Penalized LSE")
        # plt.xlabel("mu_reg")
        # plt.ylabel("lambda_reg")
        # plt.savefig("lse_sfd.png")

        # # Plot heatmap for Squared Distance (MLE)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(
        #     squared_distance_results_mle,
        #     xticklabels=[f"{x:.2g}" for x in mu_values],
        #     yticklabels=[f"{y:.2g}" for y in lambda_values],
        #     cmap="Reds",
        # )
        # plt.title("Squared Distance for Penalized MLE")
        # plt.xlabel("mu_reg")
        # plt.ylabel("lambda_reg")
        # plt.savefig("mle_dist.png")

        # # Plot heatmap for SFD Value (MLE)
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(
        #     sfd_results_mle,
        #     xticklabels=[f"{x:.2g}" for x in mu_values],
        #     yticklabels=[f"{y:.2g}" for y in lambda_values],
        #     cmap="Blues",
        # )
        # plt.title("SFD Value for Penalized MLE")
        # plt.xlabel("mu_reg")
        # plt.ylabel("lambda_reg")
        # plt.savefig("mle_sfd.png")

        print(f"Best lambda_reg for LSE: {best_lambda_lse}")
        print(f"Best mu_reg for LSE: {best_mu_lse}")
        print(f"Minimum squared distance (LSE): {min_squared_distance_lse}")

        print(f"Best lambda_reg for MLE: {best_lambda_mle}")
        print(f"Best mu_reg for MLE: {best_mu_mle}")
        print(f"Minimum squared distance (MLE): {min_squared_distance_mle}\n")

        ## Examine param values from best run:
        # best penalized LSE
        model = DevMedil(rng=rng())
        model.fit(dataset, method="lse", lambda_reg=best_lambda_lse, mu_reg=best_mu_lse)
        W_hat_lse = model.W_hat_lse
        D_hat_lse = model.D_hat_lse

        model = DevMedil(rng=rng())
        model.fit(dataset, method="mle", lambda_reg=best_lambda_mle, mu_reg=best_mu_mle)
        W_hat_mle = model.W_hat_mle
        D_hat_mle = model.D_hat_mle

        # Compute the squared distance as the evaluation metric
        with np.printoptions(precision=4, suppress=True):
            lse_order, squared_distance_lse = min_perm_squared_l2_dist_abs(
                W_hat_lse, W_star
            )
            print(
                "Squared distance between W_hat_lse and W_star:\n", squared_distance_lse
            )
            mle_order, squared_distance_mle = min_perm_squared_l2_dist_abs(
                W_hat_mle, W_star
            )
            print(
                "Squared distance between W_hat_mle and W_star:\n", squared_distance_mle
            )

            W_star = true_model.parameters.biadj_weights
            print("\nTrue weight matrix W_star:\n", W_star)
            print("Estimated W_hat (LSE):\n", W_hat_lse[np.array(lse_order)])
            print("Estimated W_hat (MLE):\n", W_hat_mle[np.array(mle_order)])

            D_star = true_model.parameters.error_variances
            print("\nTrue variances D_star:\n", D_star)
            print("Estimated variances D_hat (LSE):\n", D_hat_lse)
            print("Estimated variances D_hat (MLE):\n", D_hat_mle)


# K-Fold Cross-Validation
def benchmark_graphs_deep_dive_kfold(fixed_biadj_mat_list, k=5, verbose=False):
    for idx, biadj_matrix in enumerate(fixed_biadj_mat_list):
        print(f"\nTesting Graph {idx + 1} with shape {biadj_matrix.shape}")
        true_model = mcm(rng=rng(), parameterization="Gaussian", biadj=biadj_matrix)
        dataset = true_model.sample(5000)

        (
            validation_error_results_lse,
            validation_error_results_mle,
            squared_distance_results_lse,
            squared_distance_results_mle,
            sfd_results_lse,
            sfd_results_mle,
            mu_values,
            lambda_values,
        ) = grid_search_kfold(true_model, dataset, k=k, verbose=verbose)

        print(f"Plotting heatmaps for Graph {idx + 1}")
        plot_heatmaps(
            lambda_values,
            mu_values,
            validation_error_results_lse,
            squared_distance_results_lse,
            sfd_results_lse,
            method_name="LSE",
            fig_name=f"Graph {idx} ",
        )
        plot_heatmaps(
            lambda_values,
            mu_values,
            validation_error_results_mle,
            squared_distance_results_mle,
            sfd_results_mle,
            method_name="MLE",
            fig_name=f"Graph {idx} ",
        )


def plot_heatmaps(
    lambda_values,
    mu_values,
    validation_error_results,
    squared_distance_results,
    sfd_results,
    method_name,
    fig_name,
):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.7))
    sns.heatmap(
        validation_error_results,
        xticklabels=[f"{x:.2g}" for x in mu_values],
        yticklabels=[f"{y:.2g}" for y in lambda_values],
        square=True,
        cmap="Reds",
        ax=axs[0],
    )
    axs[0].set_title(f"{method_name} validation")
    axs[0].set_ylabel(r"$\lambda$")
    # plt.show()

    sns.heatmap(
        squared_distance_results,
        xticklabels=[f"{x:.2g}" for x in mu_values],
        yticklabels=[],
        square=True,
        cmap="Greens",
        ax=axs[1],
    )
    axs[1].set_title(f"{method_name} $l_2$")

    sns.heatmap(
        sfd_results,
        xticklabels=[f"{x:.2g}" for x in mu_values],
        yticklabels=[],
        square=True,
        cmap="Blues",
        ax=axs[2],
    )
    axs[2].set_title(f"{method_name} SFD")

    for ax in axs:
        ax.set_xlabel(r"$\mu$")

    fig.suptitle(fig_name + method_name)
    plt.savefig(fig_name + method_name + ".png")


# benchmark_graphs_deep_dive(fixed_biadj_mat_list)
benchmark_graphs_deep_dive_kfold(fixed_biadj_mat_list, k=5, verbose=True)


# add GP model code
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
from medil.independence_testing import estimate_UDG

def compute_gauss_kernel(x, sigmay, sigmax):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    n = x.shape[0]
    xnorm = squareform(pdist(x, 'euclidean')) ** 2
    kx = sigmay * np.exp(-xnorm / (2 * sigmax ** 2))
    return kx

def compute_caus_order(biadj):
    m, n = biadj.shape
    total_nodes = m + n
    remaining = list(range(total_nodes))
    caus_order = []
    adj = biadj_to_adj(biadj)
    
    while len(remaining) > 0:
        col_sums = np.sum(adj[remaining][:, remaining], axis=0)
        min_sum_index = np.argmin(col_sums)
        root = remaining[min_sum_index]
        caus_order.append(root)
        remaining.remove(root)
    
    return caus_order

def random_B(biadj):
    m, n = biadj.shape
    num_coeff = int(np.sum(biadj))
    B = biadj.T.copy()
    if num_coeff == 1:
        coeffs = np.random.uniform(0.1, 0.9) * np.random.choice([-1, 1])
    else:
        coeffs = np.random.uniform(0.1, 0.9, size=num_coeff) * np.random.choice([-1, 1], size=num_coeff)
    B[B == 1] = coeffs
    return B

def sample_data_from_G(n, biadj, func_type="GAM", noise_type="normalRandomVariances"):
    pars_func_type = {
        "B": random_B(biadj),
        "kap": 1,
        "sigmax": 1,
        "sigmay": 1,
        "output": False
    }
    pars_noise = {
        "noiseExp": 1,
        "varMin": 1,
        "varMax": 2,
        "noiseExpVarMin": 2,
        "noiseExpVarMax": 4,
        "bound": [1] * (biadj.shape[0] + biadj.shape[1])
    }

    return sample_data_from_GAM_GP(n, biadj, pars_func_type, noise_type, pars_noise)

def sample_data_from_GAM_GP(n, biadj, pars_func_type, noise_type, pars_noise):
    m, n_vars = biadj.shape
    X = np.empty((n, m+n_vars))
    caus_order = compute_caus_order(biadj)
    noise_var = np.random.uniform(pars_noise["varMin"], pars_noise["varMax"], m+n_vars)

    for node in caus_order:
        if node < m:  # Latent variable
            X[:, node] = np.random.randn(n)
        else:  # Observed variable
            pa_of_node = np.where(biadj[:, node-m] == 1)[0]
            X[:, node] = np.zeros(n)
            
            for pa in pa_of_node:
                # Non-linear relationship
                contribution = pars_func_type["B"][node-m, pa] * np.sin(X[:, pa])
                if np.any(np.isnan(contribution)) or np.any(np.isinf(contribution)):
                    print(f"Warning: Invalid values in contribution for node {node}, parent {pa}")
                    contribution = np.nan_to_num(contribution, nan=0.0, posinf=1e10, neginf=-1e10)
                X[:, node] += contribution
            
            ran = np.random.randn(n)
            noisetmp = (np.sqrt(noise_var[node]) * np.abs(ran)) ** (pars_noise["noiseExp"]) * np.sign(ran)
            X[:, node] += noisetmp
    
    return X[:, m:]

def biadj_to_adj(biadj):
    m, n = biadj.shape
    adj = np.zeros((m+n, m+n))
    adj[:m, m:] = biadj
    adj[m:, :m] = biadj.T
    return adj

def generate_dataset(biadj, n_samples, noise_scale=0.1):
    m, n_vars = biadj.shape
    X = np.empty((n_samples, n_vars))
    
    # Generate latent variables
    latent = np.random.randn(n_samples, m)
    
    # Generate observed variables
    for i in range(n_vars):
        parents = np.where(biadj[:, i] == 1)[0]
        if len(parents) == 0:
            X[:, i] = np.random.randn(n_samples)
        else:
            # Use a non-linear function to combine parent influences
            X[:, i] = np.tanh(latent[:, parents].dot(np.random.randn(len(parents))))
        
        # Add noise
        X[:, i] += noise_scale * np.random.randn(n_samples)
    
    return X

def verify_independence_pattern(biadj, n_samples=5000, noise_scale=0.1):
    # Generate data
    X = generate_dataset(biadj, n_samples, noise_scale)
    
    # Estimate UDG
    udg, p_vals = estimate_UDG(X, method='xicor')
    
    # Compute expected dependency structure
    expected_dep = (biadj.T @ biadj) > 0
    np.fill_diagonal(expected_dep, False)  # Remove self-dependencies
    
    # Compare UDG with expected dependency structure
    matches = (udg == expected_dep)
    match_rate = np.sum(matches) / (udg.shape[0] * udg.shape[1])
    
    print(f"Match rate: {match_rate:.2f}")
    print("UDG:")
    print(udg)
    print("Expected dependency structure:")
    print(expected_dep)
    print("P-values:")
    print(p_vals)
    
    return udg, expected_dep, match_rate, p_vals

# Define a verification function for GP
def verify_independence_pattern_GP(
    biadj: np.ndarray, 
    n_samples: int = 5000, 
    method: str = 'xicor'
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    parameters:
    -biadj (np.ndarray): shape = (num_latent, num_meas)
    -n_samples (int)
    -method (str):'xicor'
    
    return:
    -udg (np.ndarray)
    -expected_dep (np.ndarray)
    -match_rate (float)
    -p_vals (np.ndarray)
    """
    # Define G
    num_latent, num_meas = biadj.shape
    G = np.zeros((num_latent + num_meas, num_latent + num_meas), dtype=int)
    G[:num_latent, num_latent:] = biadj.astype(int)

    # Generate dataset
    X = sample_data_from_G(n_samples, G, func_type="GAM", noise_type="normalRandomVariances")

    # Estimate udg
    estimated = estimate_UDG(X, method=method)
    if isinstance(estimated, tuple):
        udg, p_vals = estimated
    else:
        udg = estimated
        p_vals = None  

    # Compute expected dependency structure
    expected_dep = (biadj.T @ biadj) > 0
    np.fill_diagonal(expected_dep, False)  

    print("biadj.T @ biadj:")
    print(biadj.T @ biadj)

    print("expected_dep:")
    print(expected_dep)

    # Upper triangle
    triu_indices = np.triu_indices(num_meas, k=1)
    expected_dep_triu = expected_dep[triu_indices].astype(int)
    udg_triu = udg[triu_indices].astype(int)

    # Calculte true positive, negative positive, false positive and false positive
    TP = np.sum((expected_dep_triu == 1) & (udg_triu == 1))
    TN = np.sum((expected_dep_triu == 0) & (udg_triu == 0))
    FP = np.sum((expected_dep_triu == 0) & (udg_triu == 1))
    FN = np.sum((expected_dep_triu == 1) & (udg_triu == 0))

    # Metrics
    precision = TP /(TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)

    # Match rate
    total = len(expected_dep_triu)
    match_rate = (TP + TN) / total

    # Results
    print(f"Match rate: {match_rate:.2f}")
    print("UDG:")
    print(udg_triu)
    print("Expected dependency structure:")
    print(expected_dep_triu)
    if p_vals is not None:
        print("P-values:")
        print(p_vals[triu_indices])

    print(f"True Positives: {TP}")
    print(f"True Negatives: {TN}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    return udg, expected_dep, match_rate, p_vals

# Example usage
biadj = np.array([
    [1, 1, 0],
    [0, 1, 1]
], dtype=float)

# parameters
n_samples = 5000
noise_scale = 0.1

# generate dataset
X = generate_dataset(biadj, n_samples, noise_scale)

# print part of the dataset
print(X[:10])

# verify
udg, expected_dep, match_rate, p_vals = verify_independence_pattern(biadj, n_samples=5000, noise_scale=0.1)




