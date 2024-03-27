"""MeDIL causal model base class and a preconfigured NCFA class."""
import warnings

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from scipy.optimize import minimize

from .ecc_algorithms import find_heuristic_1pc

# from learning.vae import VariationalAutoencoder
# from learning.params import train_dict
# from datetime import datetime
# import numpy as np
# import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# from learning.train import train_vae
# import torch
# import numpy as np
# import pickle
# import os

# need to decide how to organize classes; could have (gaussian)
# MedCausMod and then similar (but not inherted) NCFA;

# alternatively have MedCausMod common base and then inhert that to
# gaussian and NFCA classes?


class MedilCausalModel(object):
    def __init__(
        self,
        biadj: None | npt.NDArray = None,
        udg: None | npt.NDArray = None,
        parameterization: str = "gauss",
        one_pure_child: bool = True,
        udg_method: str = "default",
        rng=default_rng(0),
    ) -> None:
        self.biadj = biadj
        self.udg = udg
        self.parameterization = parameterization
        self.one_pure_child = one_pure_child
        self.udg_method = udg_method
        self.rng = rng
        if parameterization == "gauss":
            if self.udg_method == "default":
                self.udg_method = "bic"
            self.biadj_weights = None
            self.error_means = None
            self.error_variances = None
            # self.params = {biadj_weights: None, error_means: None,
            # error_variances: None}
        elif parameterization == "vae":
            if self.udg_method == "default":
                self.udg_method = "xicor"
            self.vae = None

    def fit(self, dataset: npt.NDArray) -> "MedilCausalModel":
        """"""
        self.dataset = dataset
        if self.biadj is None:
            self._compute_biadj()

        if self.parameterization == "gauss":
            self.error_means = self.dataset.mean(0)
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
            self.error_variances = result.x[num_weights:]
            self.biadj_weights = np.zeros_like(self.biadj, float)
            self.biadj_weights[self.biadj] = result.x[:num_weights]
            # either use scipy minimize or implement gradient descent
            # myself in numpy, or try to find more info about/how to
            # implement MLE (check MLE in Factor analysis---an
            # algebraic derivation by stoica and jansson)
        return self

    def _compute_biadj(self):
        if self.udg is None:
            self._estimate_udg()
        self.biadj = find_heuristic_1pc(self.udg)

    def _estimate_udg(self):
        if self.udg_method == "bic":
            samp_size = len(self.dataset)
            cov = np.cov(self.dataset, rowvar=False)
            corr = np.corrcoef(self.dataset, rowvar=False)
            inner_numerator = 1 - cov * corr  # should never be <= 0?
            inner_numerator = inner_numerator.clip(min=0.00001)
            inner_numerator[np.tril_indices_from(inner_numerator)] = 1
            udg_triu = np.log(inner_numerator) < (-np.log(samp_size) / samp_size)
            udg = udg_triu + udg_triu.T
        else:
            num_meas = self.dataset.shape[1]
            udg = np.ones((num_meas, num_meas), bool)
        self.udg = udg

    def sample(self, sample_size: int) -> npt.NDArray:
        if self.parameterization == "vae":
            # samp = sample drawn from vae model
            print("not implemented yet :(")
            sample = None
        elif self.parameterization == "gauss":
            num_latent = len(self.biadj)
            latent_sample = self.rng.multivariate_normal(
                np.zeros(num_latent), np.eye(num_latent), sample_size
            )
            error_sample = self.rng.multivariate_normal(
                self.error_means, np.diagflat(self.error_variances), sample_size
            )
            sample = latent_sample @ self.biadj_weights + error_sample
        return sample


class NeuroCausalFactorAnalysis(MedilCausalModel):
    def __init__(self):
        raise (NotImplementedError)

    def assign_DoF(biadj_mat, deg_of_freedom=None, method="uniform", variances=None):
        """Assign degrees of freedom (latent variables) of VAE to latent factors from causal structure learning
        Parameters
        ----------
        biadj_mat: biadjacency matrix of MCM
        deg_of_freedom: desired size of latent space of VAE
        method: how to distribute excess degrees of freedom to latent causal factors
        variances: diag of covariance matrix over measurement variables

        Returns
        -------
        redundant_biadj_mat: biadjacency matrix specifing VAE structure from latent space to decoder
        """

        num_cliques, num_obs = biadj_mat.shape
        if deg_of_freedom is None:
            # then default to upper bound; TODO: change to max_intersect_num from medil.ecc_algorithms
            deg_of_freedom = num_obs**2 // 4
        elif deg_of_freedom < num_cliques:
            warnings.warn(
                f"Input `deg_of_freedom={deg_of_freedom}` is less than the {num_cliques} required for the estimated causal structure. `deg_of_freedom` increased to {num_cliques} to compensate."
            )
            deg_of_freedom = num_cliques

        if method == "uniform":
            latents_per_clique = np.ones(num_cliques, int) * (
                deg_of_freedom // num_cliques
            )
        elif method == "clique_size":
            latents_per_clique = np.round(
                (biadj_mat.sum(1) / biadj_mat.sum()) * (deg_of_freedom - num_cliques)
            ).astype(int)
        elif method == "tot_var" or method == "avg_var":
            clique_variances = biadj_mat @ variances
            if method == "avg_var":
                clique_variances /= biadj_mat.sum(1)
            clique_variances /= clique_variances.sum()
            latents_per_clique = np.round(
                clique_variances * (deg_of_freedom - num_cliques)
            ).astype(int)

        for _ in range(2):
            remainder = deg_of_freedom - latents_per_clique.sum()
            latents_per_clique[np.argsort(latents_per_clique)[0:remainder]] += 1

        redundant_biadj_mat = np.repeat(biadj_mat, latents_per_clique, axis=0)

        return redundant_biadj_mat

    def train_vae(m, n, biadj_mat, train_loader, valid_loader, seed):
        """Training VAE with the specified image dataset
        :param m: dimension of the latent variable
        :param n: dimension of the observed variable
        :param train_loader: training image dataset loader
        :param valid_loader: validation image dataset loader
        :param biadj_mat: the adjacency matrix of the directed graph
        :param seed: random seed for the experiments
        :return: trained model and training loss history
        """

        # load parameters
        np.random.seed(seed)
        epoch, lr, beta = train_dict["epoch"], train_dict["lr"], train_dict["beta"]

        # building VAE
        mask = biadj_mat.T.astype("float32")
        mask = torch.tensor(mask).to(device)
        model = VariationalAutoencoder(m, n, mask)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.90)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Number of parameters: {num_params}"
        )

        # training loop
        model.train()
        train_elbo, train_error = [], []
        valid_elbo, valid_error = [], []

        for epoch in range(epoch):
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training on epoch {epoch}..."
            )
            train_lb, train_er, nbatch = 0.0, 0.0, 0

            for x_batch, _ in train_loader:
                batch_size = x_batch.shape[0]
                x_batch = x_batch.to(device)
                recon_batch, logcov_batch, mu_batch, logvar_batch = model(x_batch)
                loss = elbo_gaussian(
                    x_batch, recon_batch, logcov_batch, mu_batch, logvar_batch, beta
                )
                error = recon_error(x_batch, recon_batch, logcov_batch, weighted=False)
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
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish training epoch {epoch} with loss {train_lb}"
            )

            # append validation loss
            valid_lb, valid_er = valid_vae(model, valid_loader)
            valid_elbo.append(valid_lb)
            valid_error.append(valid_er)

        train_elbo, train_error = np.array(train_elbo), np.array(train_error)
        valid_elbo, valid_error = np.array(valid_elbo), np.array(valid_error)
        elbo = [train_elbo, valid_elbo]
        error = [train_error, valid_error]

        return model, elbo, error

    def valid_vae(model, valid_loader):
        """Training VAE with the specified image dataset
        :param model: trained VAE model
        :param valid_loader: validation image dataset loader
        :return: validation loss
        """

        # load parameters
        beta = train_dict["beta"]

        # set to evaluation mode
        model.eval()
        valid_lb, valid_er, nbatch = 0.0, 0.0, 0

        for x_batch, _ in valid_loader:
            with torch.no_grad():
                batch_size = x_batch.shape[0]
                x_batch = x_batch.to(device)
                recon_batch, logcov_batch, mu_batch, logvar_batch = model(x_batch)
                loss = elbo_gaussian(
                    x_batch, recon_batch, logcov_batch, mu_batch, logvar_batch, beta
                )
                error = recon_error(x_batch, recon_batch, logcov_batch, weighted=False)

                # update loss and nbatch
                valid_lb += loss.item() / batch_size
                valid_er += error.item() / batch_size
                nbatch += 1

        # report validation loss
        valid_lb = valid_lb / nbatch
        valid_er = valid_er / nbatch
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish validation with loss {valid_lb}"
        )

        return valid_lb, valid_er

    def elbo_gaussian(x, x_recon, logcov, mu, logvar, beta):
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
        cov = apply_along_axis(torch.diag, cov, axis=0)
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

    def recon_error(x, x_recon, logcov, weighted):
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
        cov = apply_along_axis(torch.diag, cov, axis=0)
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

    def apply_along_axis(function, x, axis=0):
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

    def run_vae_ablation(biadj_mat_recon, train_loader, valid_loader, path, seed):
        """Run training loop for exact VAE
        Parameters
        ----------
        biadj_mat_recon: adjacency matrix for heuristic graph
        train_loader: loader for training data
        valid_loader: loader for validation data
        path: path to save the experiments
        seed: random seed for the experiments
        """

        # train & validate heuristic MeDIL VAE
        mh, nh = biadj_mat_recon.shape
        model_recon, loss_recon, error_recon = train_vae(
            mh, nh, biadj_mat_recon, train_loader, valid_loader, seed
        )
        torch.save(model_recon, os.path.join(path, "model_recon.pt"))
        with open(os.path.join(path, "loss_recon.pkl"), "wb") as handle:
            pickle.dump(loss_recon, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, "error_recon.pkl"), "wb") as handle:
            pickle.dump(error_recon, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def pipeline_ablation(dataset, biadj_mat, dof, path, seed):
        """ablation study of DoF for fixed data set and causal structure"""

        # define paths
        path_ablation = path + f"dof={dof}_run={seed}"
        if not os.path.isdir(path):
            os.mkdir(path)

        if not os.path.isdir(path_ablation):
            os.mkdir(path_ablation)

        # load parameters
        np.random.seed(seed)

        batch_size = 251
        # batch_size = params_dict["batch_size"]
        # train_dict = {"epoch": 200, "lr": 0.005, "beta": 1}
        # params_dict = {"batch_size": 251, "num_valid": 1000}

        info = {
            "heuristic": True,
            "method": "xicor",
            "alpha": 0.05,
            "dof": dof,
            "dof_method": "uniform",
        }
        with open(os.path.join(path_ablation, "info.pkl"), "wb") as f:
            pickle.dump(info, f)

        # define VAE training and validation sample
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Preparing training and validation data for VAE"
        )
        samples, valid_samples = dataset
        train_loader = load_dataset_real(samples, batch_size)
        valid_loader = load_dataset_real(valid_samples, batch_size)

        doffed_biadj_mat = assign_DoF(biadj_mat, deg_of_freedom=dof)
        run_vae_ablation(
            doffed_biadj_mat, train_loader, valid_loader, path_ablation, seed
        )
