import time

import numpy as np
import pandas as pd
from medil import NeuroCausalFactorAnalysis as ncfa
from medil.evaluate import sfd
from torch import tensor

from eval_helper import dcor, mcc, mse

dataset = np.loadtxt(snakemake.input["dataset"])
latent_sample = np.loadtxt(snakemake.input["latent_sample"])
true_biadj = np.loadtxt(snakemake.input["graph"])
seed = int(snakemake.wildcards["seed"])
method = snakemake.wildcards["method"]

# train/val split; 70/30
rng = np.random.default_rng(seed)
n = dataset.shape[0]
perm = rng.permutation(n)
n_train = int(0.7 * n)
train_idx = perm[:n_train]
val_idx = perm[n_train:]

train = dataset[train_idx]
val = dataset[val_idx]
latent_val = latent_sample[val_idx]

# fit to the training set
model = ncfa(seed=seed)
if method == "vae":
    model._set_full_decoder_mask(num_meas=dataset.shape[1])
model.hyperparams.update(
    {
        "num_epochs": snakemake.params["num_epochs"],
        "lr": snakemake.params["lr"],
        "latent_width": snakemake.params["latent_width"],
        "meas_width": snakemake.params["meas_width"],
        "num_hidden_layers": snakemake.params["num_hidden_layers"],
        "encoder_hidden_dim": snakemake.params["encoder_hidden_dim"],
    }
)
start = time.perf_counter()
model.fit(train)
end = time.perf_counter()
mp = model.parameters

# reconstruct latent samples, E[z|x], for validation set
_mu, _logvar = mp.vae.encoder(tensor(val.astype(np.float32)))
latent_recon = _mu.detach().numpy()

# reconstruct validation data from latents
_x_recon = mp.vae.decoder(_mu)
reconstructed = _x_recon.detach().numpy()

# evaluate
result = {
    "method": [method],
    "mse": [mse(val, reconstructed)],
    "mcc": [mcc(latent_val, latent_recon)],
    "dcor": [dcor(latent_val, latent_recon)],
    "sfd": [sfd(true_biadj, model.biadj)],
    "time": end - start,
    "num_params": [sum(p.numel() for p in mp.vae.parameters() if p.requires_grad)],
}
result = pd.DataFrame(dict(snakemake.wildcards) | result)
result.to_csv(snakemake.output["result"], index=False)
np.savetxt(snakemake.output["graph"], model.biadj.astype(int), fmt="%d")
