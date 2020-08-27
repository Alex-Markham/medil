"""Example graphs and data, for use in tutorials and testing."""
import numpy as np


## Example UDGs

# Simple "M" example
simple_M_UDG = np.array(
    [
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ]
)

# Triangle example, where minECC differs from set of allmaximal cliques
triangle_UDG = np.asarray(
    [
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1],
    ]
)

# Example where number of latent variables is larger than number of measurement variables
more_latents_UDG = np.array(
    [
        [1, 1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 1],
    ]
)

# Example where multiple ECCs are possible depending on edge minimal vs vertex minimal
am_cm_diff_UDG = np.array(
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


## Corresponding example MCMs as biadjacency matriies

simple_M_cover = np.array(
    [
        [1, 1, 0],
        [0, 1, 1]
    ]
)

triangle_cover = np.array(
    [
        [1, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1]
    ]
)

more_latents_cover = np.array(
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

am_cm_diff_cover = np.array(
    [
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 1],
    ]
)


## Data
# TODO: add linear and nonlinear data example
