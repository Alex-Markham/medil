"""Example graphs and data, for use in testing and tutorials."""
import numpy as np


class ExampleUDGAndMCM(object):
    r"""Example consisting of a description, UDG, and MCM"""
    def __init__(self, description):
        self.description = description
        self.UDG = None
        self.MCM = None

    def add_udg(self, udg):
        self.UDG = np.array(udg)

    def add_mcm(self, mcm):
        self.MCM = np.array(mcm)


simple_M = ExampleUDGAndMCM("M-shaped mcm, with 2 latent- and 3 measurement-variables")
simple_M.add_udg(
    [
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ]
)
simple_M.add_mcm(
    [
        [0, 1, 1],
        [1, 1, 0]
    ]
)

triangle = ExampleUDGAndMCM("triangle example, where minECC differs from set of all maximal cliques")
triangle.add_udg(
    [
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1],
    ]
)
triangle.add_mcm(
    [
        [0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 0]
    ]
)

more_latents = ExampleUDGAndMCM("example where number of latent variables is larger than number of measurement variables")
more_latents.add_udg(
    [
        [1, 1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 1],
    ]
)
more_latents.add_mcm(
    [
        [1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
    ]
)

am_cm_diff = ExampleUDGAndMCM("example where multiple ECCs are possible depending on edge minimal vs vertex minimal")
am_cm_diff.add_udg(
    [
        [1, 1, 1, 1, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 0, 1],
        [0, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1],
    ]
)
am_cm_diff.add_mcm(
    [
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 1],
    ]
)

examples = (simple_M, triangle, more_latents, am_cm_diff)


## Data
# TODO: add linear and nonlinear data example
