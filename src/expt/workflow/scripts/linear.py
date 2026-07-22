import time

import numpy as np
import pandas as pd
from medil import GaussianMCM
from medil.evaluate import sfd

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
biadj = np.array([])  # will be learned if method not 'fa'
if method == "fa":
    num_meas = dataset.shape[1]
    biadj = np.ones((num_meas, num_meas), dtype=bool)
start = time.perf_counter()
model = GaussianMCM(biadj=biadj).fit(train)
end = time.perf_counter()
mp = model.parameters

# reconstruct latent samples, E[z|x], for validation set
Xc = val - mp.error_means
Sigma = mp.biadj_weights.T @ mp.biadj_weights + np.diag(mp.error_variances)
latent_recon = Xc @ np.linalg.solve(Sigma, mp.biadj_weights.T)

# reconstruct validation data from latents
reconstructed = latent_recon @ mp.biadj_weights + mp.error_means

# evaluate
result = {
    "method": [method],
    "mse": [mse(val, reconstructed)],
    "mcc": [mcc(latent_val, latent_recon)],
    "dcor": [dcor(latent_val, latent_recon)],
    "sfd": [sfd(true_biadj, model.biadj)],
    "time": end - start,
    "num_params": [len(mp.error_means) + len(mp.error_variances) + model.biadj.sum()],
}
result = pd.DataFrame(dict(snakemake.wildcards) | result)
result.to_csv(snakemake.output["result"], index=False)
np.savetxt(snakemake.output["graph"], model.biadj.astype(int), fmt="%d")
