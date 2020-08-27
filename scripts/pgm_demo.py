# for making sample data
import numpy as np
from medil.examples import triangle_MCM
from medil.functional_MCM import gaussian_mixture_sampler
from medil.functional_MCM import MeasurementModel  # also used in step 3

# for step 1
from medil.independence_testing import hypothesis_test, dependencies

# for step 2
from medil.ecc_algorithms import find_clique_min_cover as find_cm

# for step 3
from pytorch_lightning import Trainer
from medil.functional_MCM import uniform_sampler, GAN

# for visualization
import medil.visualize as vis


# make sample data
num_latent, num_observed = triangle_MCM.shape

decoder = MeasurementModel(biadj_mat=triangle_MCM)
sampler = gaussian_mixture_sampler(num_latent)

input_sample, output_sample = decoder.sample(sampler, num_samples=100)
# output_sample are from measurement variables


# step 1: estimate UDG
null_corr, p_vals = hypothesis_test(output_sample, num_resamples=10, measure='distance')
dep_graph = dependencies(null_corr, threshold=.1, p_vals, alpha=.1)
# dep_graph is adjacency matrix of the estimated UDG


# step 2: learn graphical MCM
learned_biadj_mat = find_cm(dep_graph)


# step 3: learn functional MCM
num_latent, num_observed = learned_biadj_mat.shape

decoder = MeasurementModel(biadj_mat=learned_biadj_mat)
sampler = data_utils.uniform_sampler(num_latent)

mmd_net = GAN(decoder, latent_sampler=sampler, batch_size=100)
mmd_net.train_on_dataset(output_sample)

trainer = Trainer(min_epochs=1000)
trainer.fit(mmd_net)


# confirm given and learned causal structures match
vis.show_graph(given_biadj_mat)
vis.show_graph(learned_biadj_mat)

# compare plots of distance correlation values for given and learned MCMs
generated_sample = decoder.sample(sampler, 1000).detach().numpy()
generated_dcor_mat = hypothesis_test(generated_sample, num_perms=1000)

vis.show_obs_dcor_mat(null_corr, print_val=True)
vis.show_obs_dcor_mat(generated_dcor_mat, print_val=True)

# get params for learned functional MCM; replace '1' with 'i' to get params for any M_i
print(decoder.observed['1'].causal_function)
