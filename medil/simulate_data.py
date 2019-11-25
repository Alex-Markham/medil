import numpy as np


class SimulatedData(object):

    def __init__(self, hmmm):
        self.samples = draw_samples(hmmm)
        self.true_model = hmmm

    def draw_samples(self):
        pass
    
