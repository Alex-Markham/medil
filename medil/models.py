"""MeDIL causal model base class and a preconfigured NCFA class."""

from datetime import datetime
import os
from pathlib import Path
import pickle
import warnings

import numpy as np
from numpy.random import default_rng
import numpy.typing as npt
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as sc
import torch
from torch.utils.data import DataLoader, TensorDataset


from .ecc_algorithms import find_heuristic_1pc
from .independence_testing import estimate_UDG
from .vae import VariationalAutoencoder


class MedilCausalModel(object):
    """Base class using principle of polymorphism to establish common
    interface for derived parametric estimators.
    """

    def __init__(
        self,
        biadj: npt.NDArray = np.array([]),
        udg: npt.NDArray = np.array([]),
        one_pure_child: bool = True,
        rng=default_rng(0),
    ) -> None:
        self.biadj = biadj
        self.udg = udg
        self.one_pure_child = one_pure_child
        self.rng = rng

    def fit(self, dataset: npt.NDArray) -> "MedilCausalModel":
        raise NotImplementedError

    def sample(self, sample_size: int) -> npt.NDArray:
        raise NotImplementedError


class Parameters(object):
    "Different parameterizations of MeDIL causal Models."

    def __init__(self, parameterization: str) -> None:
        self.parameterization = parameterization

        if parameterization == "Gaussian":
            self.error_means = np.array([])
            self.error_variances = np.array([])
            self.biadj_weights = np.array([])
        elif parameterization == "VAE":
            raise NotImplementedError

    def __str__(self) -> str:
        return "\n".join(
            f"parameters.{attr}: {val}" for attr, val in vars(self).items()
        )


class GaussianMCM(MedilCausalModel):
    """A linear MeDIL causal model with Gaussian random variables."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = Parameters("Gaussian")

    def fit(self, dataset: npt.NDArray) -> "GaussianMCM":
        """Fit a Gaussian MCM to a dataset with constraint-based
        structure learning and least squares parameter estimation."""
        self.dataset = dataset
        if self.biadj.size == 0:
            self._compute_biadj()

        self.parameters.error_means = self.dataset.mean(0)

        cov = np.cov(self.dataset, rowvar=False)

        num_weights = self.biadj.sum()
        num_err_vars = self.biadj.shape[1]

        def _objective(weights_and_err_vars):
            weights = weights_and_err_vars[:num_weights]
            err_vars = weights_and_err_vars[num_weights:]

            biadj_weights = np.zeros_like(self.biadj, float)
            biadj_weights[self.biadj] = weights

            return (
                (cov - biadj_weights.T @ biadj_weights - np.diagflat(err_vars)) ** 2
            ).sum()

        result = minimize(_objective, np.ones(num_weights + num_err_vars))
        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")

        self.parameters.error_variances = result.x[num_weights:]

        self.parameters.biadj_weights = np.zeros_like(self.biadj, float)
        self.parameters.biadj_weights[self.biadj] = result.x[:num_weights]

        return self

    def _compute_biadj(self):
        """Constraint-based structure learning."""
        if self.udg.size == 0:
            self._estimate_udg()
        self.biadj = find_heuristic_1pc(self.udg)

    def _estimate_udg(self):
        """Constraint-based structure learning."""
        samp_size = len(self.dataset)
        cov = np.cov(self.dataset, rowvar=False)
        corr = np.corrcoef(self.dataset, rowvar=False)
        inner_numerator = 1 - cov * corr  # should never be <= 0?
        inner_numerator = inner_numerator.clip(min=0.00001)
        inner_numerator[np.tril_indices_from(inner_numerator)] = 1
        udg_triu = np.log(inner_numerator) < (-np.log(samp_size) / samp_size)
        udg = udg_triu + udg_triu.T
        self.udg = udg

    def sample(self, sample_size: int) -> npt.NDArray:
        """Sample a dataset from a GaussianMCM, after structure and
        parameters have been specified or estimated."""
        num_latent = len(self.biadj)
        latent_sample = self.rng.multivariate_normal(
            np.zeros(num_latent), np.eye(num_latent), sample_size
        )
        error_sample = self.rng.multivariate_normal(
            self.parameters.error_means,
            np.diagflat(self.parameters.error_variances),
            sample_size,
        )
        sample = latent_sample @ self.parameters.biadj_weights + error_sample
        return sample


class NeuroCausalFactorAnalysis(MedilCausalModel):
    """A MeDIL causal model represented by a deep generative model."""

    def __init__(
        self,
        seed: int = 0,
        dof: int = 0,
        path: str = "trained_ncfa/",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        Path(path).mkdir(exist_ok=True)
        self.path = path
        self.verbose = verbose
        self.seed = seed
        self.hyperparams = {
            "heuristic": True,
            "method": "xicor",
            "alpha": 0.05,
            "dof": dof,
            "batch_size": 128,
            "num_epochs": 200,
            "lr": 0.005,
            "beta": 1,
            "num_valid": 1000,
        }
        self.parameters = Parameters("vae")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def log(self, entry: str) -> None:
        time_stamped_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {entry}"
        with open(f"{self.path}training.log", "a") as log_file:
            log_file.write(time_stamped_entry + "\n")
        if self.verbose:
            print(time_stamped_entry)

    def fit(self, dataset: npt.NDArray) -> "NeuroCausalFactorAnalysis":
        self.dataset = dataset
        self.doffed = self.assign_dof()

        standardized = sc().fit_transform(dataset)
        train_split, valid_split = train_test_split(
            standardized, train_size=0.7, random_state=self.seed
        )

        train_loader = self._data_loader(train_split)
        valid_loader = self._data_loader(valid_split)

        np.random.seed(self.seed)

        model_recon, loss_recon, error_recon = self._train_vae(
            train_loader, valid_loader
        )
        torch.save(model_recon, os.path.join(self.path, "model_recon.pt"))
        with open(os.path.join(self.path, "loss_recon.pkl"), "wb") as handle:
            pickle.dump(loss_recon, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.path, "error_recon.pkl"), "wb") as handle:
            pickle.dump(error_recon, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def assign_dof(self) -> npt.NDArray:
        """Assign degrees of freedom (latent variables) of VAE to
        latent factors from causal structure learning.
        """
        if self.biadj.size == 0:
            self._compute_biadj()

        num_cliques, num_meas = self.biadj.shape
        if self.hyperparams["dof"] == 0:
            # then default to 3x num_meas overcomplete
            self.dof = num_meas * 3
        elif self.dof < num_cliques:
            warnings.warn(
                f"Input `deg_of_freedom={self.dof}` is less than the {num_cliques} required for the estimated causal structure. `deg_of_freedom` increased to {num_cliques} to compensate."
            )
        self.dof = num_cliques

        latents_per_clique = np.ones(num_cliques, int) * (self.dof // num_cliques)

        for _ in range(2):
            remainder = self.dof - latents_per_clique.sum()
            latents_per_clique[np.argsort(latents_per_clique)[0:remainder]] += 1

        redundant_biadj_mat = np.repeat(self.biadj, latents_per_clique, axis=0)

        return redundant_biadj_mat

    def _compute_biadj(self):
        if self.udg.size == 0:
            self._estimate_udg()
        self.biadj = find_heuristic_1pc(self.udg)

    def _estimate_udg(self):
        self.udg, pvals = estimate_UDG(
            self.dataset,
            method=self.hyperparams["method"],
            significance_level=self.hyperparams["alpha"],
        )

    def _data_loader(self, sample):
        sample_x = sample.astype(np.float32)
        sample_z = np.empty(shape=(sample_x.shape[0], 0)).astype(np.float32)
        dataset = TensorDataset(torch.tensor(sample_x), torch.tensor(sample_z))
        data_loader = DataLoader(
            dataset, batch_size=self.hyperparams["batch_size"], shuffle=False
        )
        return data_loader

    def _train_vae(self, train_loader, valid_loader):
        """Training VAE with the specified image dataset
        :param m: dimension of the latent variable
        :param n: dimension of the observed variable
        :param train_loader: training image dataset loader
        :param valid_loader: validation image dataset loader
        :param biadj_mat: the adjacency matrix of the directed graph
        :param seed: random seed for the experiments
        :return: trained model and training loss history
        """

        m, n = self.doffed.shape

        # building VAE
        mask = self.doffed.T.astype("float32")
        mask = torch.tensor(mask).to(self.device)
        model = VariationalAutoencoder(m, n, mask)
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.hyperparams["lr"], weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.90)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log(f"Number of parameters: {num_params}")

        # training loop
        model.train()
        train_elbo, train_error = [], []
        valid_elbo, valid_error = [], []

        for idx in range(self.hyperparams["num_epochs"]):
            self.log(f"Training on epoch {idx}...")
            train_lb, train_er, nbatch = 0.0, 0.0, 0

            for x_batch, _ in train_loader:
                batch_size = x_batch.shape[0]
                x_batch = x_batch.to(self.device)
                recon_batch, logcov_batch, mu_batch, logvar_batch = model(x_batch)
                loss = self._elbo_gaussian(
                    x_batch,
                    recon_batch,
                    logcov_batch,
                    mu_batch,
                    logvar_batch,
                    self.hyperparams["beta"],
                )
                error = self._recon_error(
                    x_batch, recon_batch, logcov_batch, weighted=False
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update loss and nbatch
                train_lb += loss.item() / batch_size
                train_er += error.item() / batch_size
                nbatch += 1

            # finish training epoch
            scheduler.step()
            train_lb = train_lb / nbatch
            train_er = train_er / nbatch
            train_elbo.append(train_lb)
            train_error.append(train_er)
            self.log(f"Finish training epoch {idx} with loss {train_lb}")

            # append validation loss
            valid_lb, valid_er = self._valid_vae(model, valid_loader)
            valid_elbo.append(valid_lb)
            valid_error.append(valid_er)

        train_elbo, train_error = np.array(train_elbo), np.array(train_error)
        valid_elbo, valid_error = np.array(valid_elbo), np.array(valid_error)
        elbo = [train_elbo, valid_elbo]
        error = [train_error, valid_error]

        return model, elbo, error

    def _valid_vae(self, model, valid_loader):
        """Training VAE with the specified image dataset
        :param model: trained VAE model
        :param valid_loader: validation image dataset loader
        :return: validation loss
        """
        # set to evaluation mode
        model.eval()
        valid_lb, valid_er, nbatch = 0.0, 0.0, 0

        for x_batch, _ in valid_loader:
            with torch.no_grad():
                batch_size = x_batch.shape[0]
                x_batch = x_batch.to(self.device)
                recon_batch, logcov_batch, mu_batch, logvar_batch = model(x_batch)
                loss = self._elbo_gaussian(
                    x_batch,
                    recon_batch,
                    logcov_batch,
                    mu_batch,
                    logvar_batch,
                    self.hyperparams["beta"],
                )
                error = self._recon_error(
                    x_batch, recon_batch, logcov_batch, weighted=False
                )

                # update loss and nbatch
                valid_lb += loss.item() / batch_size
                valid_er += error.item() / batch_size
                nbatch += 1

        # report validation loss
        valid_lb = valid_lb / nbatch
        valid_er = valid_er / nbatch
        self.log(f"Finish validation with loss {valid_lb}")

        return valid_lb, valid_er

    def _elbo_gaussian(self, x, x_recon, logcov, mu, logvar, beta):
        """Calculating loss for variational autoencoder
        :param x: original image
        :param x_recon: reconstruction in the output layer
        :param logcov: log of covariance matrix of the data distribution
        :param mu: mean in the fitted variational distribution
        :param logvar: log of the variance in the variational distribution
        :param beta: beta
        :return: reconstruction loss + KL
        """

        # KL-divergence
        # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        # https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
        # https://arxiv.org/pdf/1312.6114.pdf
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # reconstruction loss
        cov = torch.exp(logcov)
        cov = self._apply_along_axis(torch.diag, cov, axis=0)
        cov = cov.mean(axis=0)

        diff = x - x_recon
        recon_loss = torch.sum(
            torch.det(cov)
            + torch.diagonal(
                torch.mm(
                    torch.mm(diff, torch.inverse(cov)), torch.transpose(diff, 0, 1)
                )
            )
        ).mul(-1 / 2)

        # elbo
        loss = -beta * kl_div + recon_loss

        return -loss

    def _recon_error(self, x, x_recon, logcov, weighted):
        """Reconstruction error given x and x_recon
        :param x: original image
        :param x_recon: reconstruction in the output layer
        :param logcov: covariance matrix of the data distribution
        :param weighted: whether to use weighted reconstruction

        Returns
        -------
        error: reconstruction error
        """

        # reconstruction loss
        cov = torch.exp(logcov)
        cov = self._apply_along_axis(torch.diag, cov, axis=0)
        cov = cov.mean(axis=0)

        diff = x - x_recon
        if weighted:
            error = torch.sum(
                torch.det(cov)
                + torch.diagonal(
                    torch.mm(
                        torch.mm(diff, torch.inverse(cov)), torch.transpose(diff, 0, 1)
                    )
                )
            ).mul(-1 / 2)
        else:
            error = torch.linalg.norm(diff, ord=2)

        return error

    @staticmethod
    def _apply_along_axis(function, x, axis=0):
        """Helper function to return along a particular axis
        Parameters
        ----------
        function: function to be applied
        x: data
        axis: axis to apply the function

        Returns
        -------
        The output applied to the axis
        """

        return torch.stack(
            [function(x_i) for x_i in torch.unbind(x, dim=axis)], dim=axis
        )


# implement penalized mle and penalized lse with a new class
class DevMedil(MedilCausalModel):
    def __init__(
        self,
        biadj: npt.NDArray = np.array([]),
        udg: npt.NDArray = np.array([]),
        one_pure_child: bool = True,
        rng=np.random.default_rng(0),
        lambda_reg: float = 0.1,
        mu_reg: float = 0.1,
    ) -> None:
        super().__init__(biadj, udg, one_pure_child, rng)
        self.lambda_reg = lambda_reg
        self.mu_reg = mu_reg

    # penalized MLE
    def fit_penalized_mle(self, dataset: npt.NDArray) -> "DevMedil":
        k, n = self.biadj.shape
        Sigma_hat = np.cov(dataset.T)

        def penalized_mle_loss(params):
            W = params[: k * n].reshape(k, n)
            D = np.diag(params[k * n :])

            Sigma = self.compute_sigma(W, D)
            Sigma_inv = np.linalg.inv(Sigma)
            sign, logdet = np.linalg.slogdet(Sigma_inv)

            if sign <= 0:
                return np.inf

            loss = np.trace(np.dot(Sigma_hat, Sigma_inv)) - logdet
            loss += self.lambda_reg * self.rho(W) + self.mu_reg * self.sigma(W)

            return loss

        initial_W = self.rng.standard_normal((k, n))
        initial_D = self.rng.random(n)
        initial_params = np.hstack([initial_W.flatten(), initial_D])

        result = minimize(penalized_mle_loss, initial_params, method="BFGS")
        self.result = result
        self.W_hat_mle = result.x[: k * n].reshape(k, n)
        self.D_hat_mle = np.diag(result.x[k * n :])
        self.convergence_success_mle = result.success
        self.convergence_message_mle = result.message

        return self

    # penalized LSE
    def fit_penalized_lse(self, dataset: npt.NDArray) -> "DevMedil":
        k, n = self.biadj.shape
        Sigma_hat = np.cov(dataset.T)

        def penalized_lse_loss(params):
            W = params[: k * n].reshape(k, n)
            D = np.diag(params[k * n :])

            loss = norm(Sigma_hat - W.T @ W - D, "fro") ** 2
            # nuclear norm for the first penalty term
            loss += self.lambda_reg * self.rho(W)
            # L1 norm for the second penalty function
            loss += self.mu_reg * self.sigma(W)

            return loss

        initial_W = self.rng.standard_normal((k, n))
        initial_D = np.abs(self.rng.standard_normal(n))
        initial_params = np.concatenate([initial_W.flatten(), initial_D])

        result = minimize(penalized_lse_loss, initial_params, method="BFGS")
        self.result = result
        self.W_hat_lse = result.x[: k * n].reshape(k, n)
        self.D_hat_lse = np.diag(np.abs(result.x[k * n :]))
        self.convergence_success_lse = result.success
        self.convergence_message_lse = result.message

        return self

    # compute sigma
    def compute_sigma(self, W: npt.NDArray, D: npt.NDArray) -> npt.NDArray:
        Sigma = np.dot(W.T, W) + D
        return Sigma

    # ρ(W), the nuclear norm defined as tr(√(W⊤W))
    def rho(self, W: npt.NDArray) -> float:
        WWt = np.dot(W.T, W)  # Compute W^T times W
        sqrt_WWt = sqrtm(WWt)  # Compute the matrix square root of W^T times W
        sqrt_WWt = np.real(sqrt_WWt)  # Take only the real part
        return np.trace(sqrt_WWt)  # Take the trace of the resulting matrix

    # σ(W), the sum of absolute values of elements (L1 norm)
    def sigma(self, W: npt.NDArray) -> float:
        return np.sum(np.abs(W))

    def sample(self, sample_size: int, method: str = "mle") -> npt.NDArray:
        if method not in ["mle", "lse"]:
            raise ValueError("Method must be either 'mle' or 'lse'")

        if method == "mle":
            if not hasattr(self, "W_hat_mle") or not hasattr(self, "D_hat_mle"):
                raise ValueError("MLE model must be fitted before sampling")
            W_hat, D_hat = self.W_hat_mle, self.D_hat_mle
        else:
            if not hasattr(self, "W_hat_lse") or not hasattr(self, "D_hat_lse"):
                raise ValueError("LSE model must be fitted before sampling")
            W_hat, D_hat = self.W_hat_lse, self.D_hat_lse

        k, n = W_hat.shape
        L = self.rng.standard_normal((sample_size, k))
        epsilon = self.rng.multivariate_normal(np.zeros(n), D_hat, sample_size)
        return np.dot(L, W_hat) + epsilon

    def fit(self, dataset: npt.NDArray, method: str = "mle") -> "DevMedil":
        if method == "mle":
            return self.fit_penalized_mle(dataset)
        elif method == "lse":
            return self.fit_penalized_lse(dataset)
        else:
            raise ValueError("Method must be either 'mle' or 'lse'")
