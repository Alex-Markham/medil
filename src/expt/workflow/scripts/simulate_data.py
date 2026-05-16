import numpy as np
from medil import sample
from numpy.random import default_rng

num_latent = int(snakemake.params["num_latent"])
num_meas = int(snakemake.params["num_meas"])
edge_prob = float(snakemake.params["edge_prob"])
samp_size = int(snakemake.params["samp_size"])
seed = int(snakemake.wildcards["seed"])

model = sample.mcm(
    num_meas=num_meas, num_latent=num_latent, density=edge_prob, rng=default_rng(seed)
)
dataset, latent_sample = model.sample(samp_size, True)

np.savetxt(snakemake.output["dataset"], dataset)
np.savetxt(snakemake.output["latent_sample"], latent_sample)
np.savetxt(snakemake.output["graph"], model.biadj.astype(int), fmt="%d")
