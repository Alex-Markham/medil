from learning.train import train_vae
import torch
import numpy as np
import pickle
import os


def run_vae_oracle(biadj_mat, train_loader, valid_loader, cov_train, cov_valid, path, seed):
    """ Run training loop for oracle VAE
    Parameters
    ----------
    biadj_mat: ground truth adjacency matrix
    train_loader: loader for training data
    valid_loader: loader for validation data
    cov_train: covariance matrix for training data
    cov_valid: covariance matrix for validation data
    path: path to save the experiments
    seed: random seed for the experiments
    """

    # train & validate oracle VAE
    m, n = biadj_mat.shape
    model_true, loss_true, error_true = train_vae(
        m, n, biadj_mat, train_loader, valid_loader, cov_train, cov_valid, seed
    )
    torch.save(model_true, os.path.join(path, "model_true.pt"))
    with open(os.path.join(path, "loss_true.pkl"), "wb") as handle:
        pickle.dump(loss_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_true.pkl"), "wb") as handle:
        pickle.dump(error_true, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_vae_suite(biadj_mat_recon, train_loader, valid_loader, cov_train, cov_valid, path, seed):
    """ Run training loop for exact VAE
    Parameters
    ----------
    biadj_mat_recon: adjacency matrix for heuristic graph
    train_loader: loader for training data
    valid_loader: loader for validation data
    cov_train: covariance matrix for training data
    cov_valid: covariance matrix for validation data
    path: path to save the experiments
    seed: random seed for the experiments
    """

    # train & validate heuristic MeDIL VAE
    mh, nh = biadj_mat_recon.shape
    model_recon, loss_recon, error_recon = train_vae(
        mh, nh, biadj_mat_recon, train_loader, valid_loader, cov_train, cov_valid, seed
    )
    torch.save(model_recon, os.path.join(path, "model_recon.pt"))
    with open(os.path.join(path, "loss_recon.pkl"), "wb") as handle:
        pickle.dump(loss_recon, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_recon.pkl"), "wb") as handle:
        pickle.dump(error_recon, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # train & validate vanilla VAE
    biadj_mat_vanilla = np.ones((mh, nh))
    model_vanilla, loss_vanilla, error_vanilla = train_vae(
        mh, nh, biadj_mat_vanilla, train_loader, valid_loader, cov_train, cov_valid, seed
    )
    torch.save(model_vanilla, os.path.join(path, "model_vanilla.pt"))
    with open(os.path.join(path, "loss_vanilla.pkl"), "wb") as handle:
        pickle.dump(loss_vanilla, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_vanilla.pkl"), "wb") as handle:
        pickle.dump(error_vanilla, handle, protocol=pickle.HIGHEST_PROTOCOL)
