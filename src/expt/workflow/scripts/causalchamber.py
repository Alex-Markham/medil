import os

import causalchamber.datasets as chamber
import numpy as np

chamber_root = os.path.dirname(snakemake.output["dataset"])

os.makedirs(chamber_root, exist_ok=True)

expt = "lt_interventions_standard_v1"
dataset = chamber.Dataset(name=expt, root=chamber_root, download=True)
experiment = dataset.get_experiment(name="uniform_reference")

df = experiment.as_pandas_dataframe()

meas = [
    "current",
    "angle_1",
    "angle_2",
    "ir_1",
    "ir_2",
    "ir_3",
    "vis_3",
]
latent = [
    "pol_1",
    "pol_2",
    "red",
    "green",
    "blue",
    "l_11",
    "l_12",
    "l_21",
    "l_22",
    "l_31",
    "l_32",
]

graph = np.array(
    [
        [1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 1],
    ]
)

dataset = df[meas].to_numpy()
dataset = (dataset - dataset.mean(0)) / dataset.std(0)

latent_sample = df[latent].to_numpy()
latent_sample = (latent_sample - latent_sample.mean(0)) / latent_sample.std(0)

np.savetxt(snakemake.output["dataset"], dataset)
np.savetxt(snakemake.output["latent_sample"], latent_sample)
np.savetxt(snakemake.output["graph"], graph, fmt="%d")
