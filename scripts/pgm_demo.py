# for making sample data
import numpy as np
from medil.examples import triangle_MCM
from medil.functional_MCM import gaussian_mixture_sampler
from medil.functional_MCM import MeDILCausalModel  # also used in step 3

# for step 1
from medil.independence_testing import hypothesis_test, dependencies, distance

# for step 2
from medil.ecc_algorithms import find_clique_min_cover as find_cm

# for step 3
from pytorch_lightning import Trainer
from medil.functional_MCM import uniform_sampler, GAN

# for visualization
import medil.visualize as vis


# make sample data
num_latent, num_observed = triangle_MCM.shape

decoder = MeDILCausalModel(biadj_mat=triangle_MCM)
sampler = gaussian_mixture_sampler(num_latent)

input_sample, output_sample = decoder.sample(sampler, num_samples=10000)
np.save('measurement_data', output_sample)

# step 1: estimate UDG
p_vals, null_corr = hypothesis_test(output_sample.T, num_resamples=100, measure='distance')
dep_graph = dependencies(null_corr, .1, p_vals, .1)
# dep_graph is adjacency matrix of the estimated UDG


# step 2: learn graphical MCM
learned_biadj_mat = find_cm(dep_graph)


# step 3: learn functional MCM
num_latent, num_observed = learned_biadj_mat.shape

decoder = MeDILCausalModel(biadj_mat=learned_biadj_mat)
sampler = uniform_sampler(num_latent)

minMCM = GAN('measurement_data.npy', decoder, latent_sampler=sampler, batch_size=100)
trainer = Trainer(min_epochs=1000)
trainer.fit(minMCM)


# confirm given and learned causal structures match
vis.show_dag(triangle_MCM)
vis.show_dag(learned_biadj_mat)

# compare plots of distance correlation values for given and learned MCMs
generated_sample = decoder.sample(sampler, 1000)[1].detach().numpy()
generated_dcor_mat = distance(generated_sample.T)

vis.show_obs_dcor_mat(null_corr, print_val=True)
vis.show_obs_dcor_mat(generated_dcor_mat, print_val=True)

# get params for learned functional MCM; replace '0' with 'i' to get params for any M_i
print(decoder.observed['0'].causal_function)
