"""Example graphs and data, for use in testing and tutorials."""
import numpy as np


def simple_M_UDG():
    return np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1]
        ]
    )


# Triangle example, where minECC differs from set of allmaximal cliques
def triangle_UDG():
    return np.asarray(
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
def more_latents_UDG():
    return np.array(
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
def am_cm_diff_UDG():
    return np.array(
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


def simple_M_MCM():
    return np.array(
        [
            [0, 1, 1],
            [1, 1, 0]
        ]
    )


def triangle_MCM():
    return np.array(
        [
            [0, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 0, 0]
        ]
    )


def more_latents_MCM():
    return np.array(
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


def am_cm_diff_MCM():
    return np.array(
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
