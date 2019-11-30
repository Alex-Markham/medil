"""Various kinds of simulated data

This currently only includes a simple linear Gaussian additive MCM method for now.

"""
import numpy as np


class SimulatedData(object):

    def __init__(self, num_samples, num_measured, num_latents, num_diff_FCMs=1, prob_of_FCMs=None):
        self.num_samples = num_samples
        self.num_measured = num_measured
        self.num_latents = num_latents
        self.num_diff_FCMs = num_diff_FCMs
        self.prob_of_FCMs = np.array(prob_of_FCMs)
        self.true_FCMs = make_random_FCMs()
        self.samples = draw_samples()

    def make_random_FCMs(self):
        true_FCMs = np.zeros((num_measured, num_latents, num_diff_FCMs))
        # now randomly make mask
        
        # true_FCMs = dict()
        # for idx in range(self.num_diff_FCMs):
            
        #     true_FCMs['model_' + str(idx)] =
        return true_FCMs

        
    def draw_samples(self):
        pass
    
