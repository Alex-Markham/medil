"""MeDIL causal model classes for linear Gaussian and deep generative settings."""

import copy
import os
import pickle
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from ._ecc_algorithms import _find_heuristic_1pc
from ._independence_testing import _estimate_UDG

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    from ._vae import VariationalAutoencoder
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _MedilCausalModel(object):
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

    def fit(self, dataset: npt.NDArray) -> "_MedilCausalModel":
        raise NotImplementedError

    def sample(self, sample_size: int) -> npt.NDArray:
        raise NotImplementedError


class _Parameters(object):
    "Different parameterizations of MeDIL causal Models."

    def __init__(self, parameterization: str) -> None:
        self.parameterization = parameterization

        if parameterization == "Gaussian":
            self.error_means = np.array([])
            self.error_variances = np.array([])
            self.biadj_weights = np.array([])
        elif parameterization == "VAE":
            self.weights = np.array([])
            self.vae = None

    def __str__(self) -> str:
        return "\n".join(
            f"parameters.{attr}: {val}" for attr, val in vars(self).items()
        )


class GaussianMCM(_MedilCausalModel):
    """Linear Gaussian MeDIL causal model.

    Learns a bipartite latent→measurement causal structure and estimates
    linear Gaussian parameters (edge weights, error means, error variances)
    by constraint-based structure learning and least-squares optimization
    of the covariance matrix.

    Parameters
    ----------
    biadj : ndarray of shape (num_latent, num_meas), optional
        Boolean biadjacency matrix. If empty (default), estimated from data
        during :meth:`fit`.
    udg : ndarray of shape (num_meas, num_meas), optional
        Boolean undirected dependence graph over observed variables. If empty
        (default), estimated from data during :meth:`fit`.
    rng : numpy.random.Generator, optional
        Random number generator used during :meth:`sample`.

    Attributes
    ----------
    biadj : ndarray of shape (num_latent, num_meas)
        Boolean biadjacency matrix (set after :meth:`fit` or at init).
    parameters : object
        Learned parameters with attributes ``biadj_weights``
        (shape ``(num_latent, num_meas)``), ``error_means``
        (shape ``(num_meas,)``), and ``error_variances``
        (shape ``(num_meas,)``).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = _Parameters("Gaussian")

    def fit(self, dataset: npt.NDArray) -> "GaussianMCM":
        """Fit a GaussianMCM to a dataset.

        Estimates the biadjacency matrix via constraint-based structure
        learning (if not pre-specified), then estimates edge weights and
        error variances by least-squares optimization of the covariance.

        Parameters
        ----------
        dataset : ndarray of shape (n_samples, n_features)
            Observed data matrix. Rows are observations, columns are variables.

        Returns
        -------
        self : GaussianMCM
        """
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
        self.biadj = _find_heuristic_1pc(self.udg)

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

    def sample(self, sample_size: int, include_latent: bool = False) -> npt.NDArray:
        """Sample observations from a GaussianMCM.

        Requires ``biadj`` and ``parameters`` to be set, either by calling
        :meth:`fit` or by constructing the model via :func:`medil.sample.mcm`.

        Parameters
        ----------
        sample_size : int
            Number of observations to draw.
        include_latent : bool, optional
            If True, also return the sampled latent variables.

        Returns
        -------
        samples : ndarray of shape (sample_size, num_meas)
        latent_samples : ndarray of shape (sample_size, num_latent), only if include_latent=True
        """
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

        return (sample, latent_sample) if include_latent else sample


class NeuroCausalFactorAnalysis(_MedilCausalModel):
    """Nonlinear MeDIL causal model represented by a masked variational autoencoder.

    Jointly learns the causal factor structure (``biadj``) and nonlinear
    generative mechanisms via a masked VAE whose decoder connectivity encodes
    the latent→measurement graph :cite:`markham2023neuro`.

    Requires PyTorch: ``pip install medil[ncfa]``.

    Input data should be standardized (zero mean, unit variance per feature)
    before calling :meth:`fit`.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Default 0.
    log_path : str, optional
        Directory for training artifacts (model weights, loss history).
        Created if it does not exist. No artifacts written if empty (default).
    verbose : bool, optional
        Print timestamped training log entries to stdout. Default False.
    biadj : ndarray of shape (num_latent, num_meas), optional
        Boolean biadjacency matrix. If empty (default), estimated from data
        during :meth:`fit` using xi correlation.
    **kwargs
        Additional keyword arguments passed to the base class (``udg``, ``rng``).

    Attributes
    ----------
    biadj : ndarray of shape (num_latent, num_meas)
        Boolean biadjacency matrix (set after :meth:`fit` or at init).
    parameters : object
        Learned VAE, accessible as ``parameters.vae``.
    loss : dict or None
        Train/validation ELBO and reconstruction losses after :meth:`fit`,
        keyed by ``"elbo_train"``, ``"elbo_valid"``, ``"recon_train"``,
        ``"recon_valid"``.
    hyperparams : dict
        Training hyperparameters. Modify via ``model.hyperparams.update({...})``
        before calling :meth:`fit`. Keys:

        - ``"method"`` : independence test for structure learning
          (``"xicor"`` or ``"dcov_fast"``; default ``"xicor"``)
        - ``"alpha"`` : significance level for independence tests (default 0.05)
        - ``"num_epochs"`` : maximum training epochs (default 200)
        - ``"lr"`` : AdamW learning rate (default 1e-3)
        - ``"beta"`` : KL weight in the ELBO (default 1.0)
        - ``"latent_width"`` : hidden units per latent variable in the decoder (default 2)
        - ``"meas_width"`` : hidden units per measurement variable in the decoder (default 2)
        - ``"num_hidden_layers"`` : number of decoder hidden layers (default 1)
        - ``"encoder_hidden_dim"`` : hidden dimension of the encoder MLP (default 64)
        - ``"batch_size"`` : mini-batch size (default 128)
        - ``"early_stopping"`` : stop when validation ELBO stagnates (default True)
        - ``"patience"`` : early stopping patience in epochs (default 20)
        - ``"min_delta"`` : minimum ELBO improvement to reset patience (default 1e-4)
    """

    def __init__(
        self,
        seed: int = 0,
        log_path: str = "",
        verbose: bool = False,
        **kwargs,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "NeuroCausalFactorAnalysis requires PyTorch. "
                "Install it with: pip install medil[ncfa]"
            )
        super().__init__(**kwargs)

        if log_path:
            Path(log_path).mkdir(exist_ok=True)

        self.log_path = log_path
        self.verbose = verbose
        self.seed = seed

        self.hyperparams = {
            "method": "xicor",
            "alpha": 0.05,
            "batch_size": 128,
            "num_epochs": 200,
            "lr": 1e-3,
            "beta": 1.0,
            "latent_width": 2,
            "meas_width": 2,
            "num_hidden_layers": 1,
            "encoder_hidden_dim": 64,
            "shuffle": True,
            "early_stopping": True,
            "patience": 20,
            "min_delta": 1e-4,
        }

        self.parameters = _Parameters("VAE")
        self.loss = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _log(self, entry: str) -> None:
        if not (self.log_path or self.verbose):
            return

        time_stamped_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {entry}"

        if self.log_path:
            with open(os.path.join(self.log_path, "training.log"), "a") as log_file:
                log_file.write(time_stamped_entry + "\n")

        if self.verbose:
            print(time_stamped_entry)

    def _set_deterministic_seed(self):
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def fit(self, dataset: npt.NDArray, split_idcs=None) -> "NeuroCausalFactorAnalysis":
        """Fit a NeuroCausalFactorAnalysis model to a dataset using a masked VAE.

        Parameters
        ----------
        dataset : ndarray of shape (n_samples, n_features)
            Observed data matrix. Should be standardized (zero mean, unit
            variance per feature) for best results.
        split_idcs : tuple of index arrays, optional
            (train_indices, valid_indices). If None, a 70/30 train/valid split
            is created automatically using self.seed.

        Returns
        -------
        self : NeuroCausalFactorAnalysis
        """
        self._set_deterministic_seed()
        self.dataset = dataset

        if self.biadj.size == 0:
            self._compute_biadj()

        if split_idcs is None:
            train_split, valid_split = train_test_split(
                dataset, train_size=0.7, random_state=self.seed
            )
        else:
            train_split = dataset[split_idcs[0]]
            valid_split = dataset[split_idcs[1]]

        train_loader = self._data_loader(train_split)
        valid_loader = self._data_loader(valid_split)

        model_recon, loss_recon, error_recon = self._train_vae(
            train_loader, valid_loader
        )

        if self.log_path:
            torch.save(
                model_recon.state_dict(), os.path.join(self.log_path, "model_recon.pt")
            )
            with open(os.path.join(self.log_path, "loss_recon.pkl"), "wb") as handle:
                pickle.dump(loss_recon, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(self.log_path, "error_recon.pkl"), "wb") as handle:
                pickle.dump(error_recon, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.parameters.vae = model_recon
        self.loss = {
            "elbo_train": loss_recon[0],
            "elbo_valid": loss_recon[1],
            "recon_train": error_recon[0],
            "recon_valid": error_recon[1],
        }

        return self

    def _compute_biadj(self):
        if self.udg.size == 0:
            self._estimate_udg()
        self.biadj = _find_heuristic_1pc(self.udg)

    def _estimate_udg(self):
        self.udg, _ = _estimate_UDG(
            self.dataset,
            method=self.hyperparams["method"],
            significance_level=self.hyperparams["alpha"],
        )

    def _data_loader(self, sample):
        sample_x = sample.astype(np.float32)
        dataset = TensorDataset(torch.tensor(sample_x))
        return DataLoader(
            dataset,
            batch_size=self.hyperparams["batch_size"],
            shuffle=self.hyperparams["shuffle"],
            num_workers=0,
        )

    def _train_vae(self, train_loader, valid_loader):
        num_meas = self.dataset.shape[1]
        biadj = torch.tensor(self.biadj.T, dtype=torch.float32)

        model = VariationalAutoencoder(
            num_latent=biadj.shape[1],
            num_meas=num_meas,
            num_hidden_layers=self.hyperparams["num_hidden_layers"],
            latent_width=self.hyperparams["latent_width"],
            meas_width=self.hyperparams["meas_width"],
            biadj=biadj,
            encoder_hidden_dim=self.hyperparams["encoder_hidden_dim"],
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.hyperparams["lr"])

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log(f"Number of parameters: {num_params}")

        train_elbo, train_error = [], []
        valid_elbo, valid_error = [], []

        best_valid = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        pbar = tqdm(
            range(self.hyperparams["num_epochs"]), desc="Training NCFA", unit="epoch"
        )

        for epoch in pbar:
            model.train()

            for (x_batch,) in train_loader:
                x_batch = x_batch.to(self.device)

                x_recon, mu, logvar = model(x_batch)
                loss = self._vae_loss(
                    x_batch, x_recon, mu, logvar, beta=self.hyperparams["beta"]
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_lb, train_er = self._eval_loss(model, train_loader)
            train_elbo.append(train_lb)
            train_error.append(train_er)

            valid_lb, valid_er = self._eval_loss(model, valid_loader)
            valid_elbo.append(valid_lb)
            valid_error.append(valid_er)

            pbar.set_postfix({"train": train_lb, "valid": valid_lb})

            improved = valid_lb < (best_valid - self.hyperparams["min_delta"])
            if improved:
                best_valid = valid_lb
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                self.hyperparams["early_stopping"]
                and epochs_without_improvement >= self.hyperparams["patience"]
            ):
                self._log(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(best_state)

        return (
            model,
            [np.array(train_elbo), np.array(valid_elbo)],
            [np.array(train_error), np.array(valid_error)],
        )

    def _eval_loss(self, model, loader):
        model.eval()
        total_loss = 0.0
        total_recon = 0.0
        n = 0

        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                x_recon, mu, logvar = model(x_batch)
                loss = self._vae_loss(
                    x_batch, x_recon, mu, logvar, beta=self.hyperparams["beta"]
                )
                recon = self._recon_error(x_batch, x_recon)

                bs = x_batch.shape[0]
                total_loss += loss.item()
                total_recon += recon.item()
                n += bs

        return total_loss / n, total_recon / n

    @staticmethod
    def _vae_loss(x, x_recon, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_div

    @staticmethod
    def _recon_error(x, x_recon):
        return torch.linalg.norm(x - x_recon, ord=2)

    def sample(self, sample_size: int, include_latent: bool = False) -> npt.NDArray:
        """Sample observations from a fitted NeuroCausalFactorAnalysis model.

        Parameters
        ----------
        sample_size : int
            Number of samples to generate.
        include_latent : bool, optional
            If True, also return the latent codes z drawn from the prior.
            Note: z has shape (sample_size, num_latent * latent_width).

        Returns
        -------
        sample : ndarray of shape (sample_size, num_meas)
        latent_sample : ndarray of shape (sample_size, latent_dim), only if include_latent=True
        """
        if self.parameters.vae is None:
            raise ValueError("Model must be fitted before sampling.")
        vae = self.parameters.vae
        latent_dim = vae.decoder.latent_dim
        z = torch.randn(sample_size, latent_dim, device=self.device)
        with torch.no_grad():
            vae.eval()
            x_recon = vae.decoder(z)
        out = x_recon.cpu().numpy()
        return (out, z.cpu().numpy()) if include_latent else out

    def _set_full_decoder_mask(self, num_meas=None):
        if num_meas is None:
            if not hasattr(self, "dataset"):
                raise ValueError("Provide num_meas or set dataset first.")
            num_meas = self.dataset.shape[1]

        num_latent = num_meas
        self.biadj = np.ones((num_latent, num_meas), dtype=float)
