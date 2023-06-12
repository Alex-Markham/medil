"""Randomly sample from and generate functional MeDIL Causal Models."""
import warnings

from numpy.random import default_rng
import numpy as np
from gues.grues import InputData as rand_walker

from .ecc_algorithms import find_heuristic_clique_cover as find_h
from .ecc_algorithms import find_clique_min_cover as find_cm


def rand_biadj_mat(num_obs, edge_prob, rng=default_rng(0)):
    """Generate random undirected graph over observed variables
    Parameters
    ----------
    num_obs: dimension of the observed space
    edge_prob: edge probability
    rng: random generator

    Returns
    -------
    biadj_mat: biadjacency matrix of the directed graph, with entry (i,j) indicating an edge from latent variable L_i to measurement variable M_j
    """

    udg = np.zeros((num_obs, num_obs), bool)
    max_edges = (num_obs * (num_obs - 1)) // 2
    num_edges = np.round(edge_prob * max_edges).astype(int)
    edges = np.ones(max_edges)
    edges[num_edges:] = 0
    udg[np.triu_indices(num_obs, k=1)] = rng.permutation(edges)
    udg += udg.T
    np.fill_diagonal(udg, True)

    # find latent connections (minimum edge clique cover)
    biadj_mat = find_h(udg)
    biadj_mat = biadj_mat.astype(bool)

    return biadj_mat


def sample_from_minMCM(minMCM, num_samps=1000, rng=default_rng(0)):
    """Sample from the minMCM graph: minMCM should either be the binary bi-adjacency matrix or covariance matrix
    Parameters
    ----------
    minMCM: covariance matrix over full MCM or boolean biadjacency matrix of MCM (for which random cov matrix will be generated)
    num_samps: number of samples
    rng: random generator

    Returns
    -------
    samples: samples
    cov: covariance matrix
    """

    if minMCM.dtype == bool:
        biadj_mat = minMCM

        # generate random weights in +-[0.5, 2]
        num_edges = biadj_mat.sum()
        num_latent, num_obs = biadj_mat.shape
        idcs = np.argwhere(biadj_mat)
        idcs[:, 1] += num_latent

        weights = (rng.random(num_edges) * 1.5) + 0.5
        weights[rng.choice((True, False), num_edges)] *= -1

        precision = np.eye(num_latent + num_obs, dtype=float)
        precision[idcs[:, 0], idcs[:, 1]] = weights
        precision = precision.dot(precision.T)

        cov = np.linalg.inv(precision)

    else:
        cov = minMCM

    samples = rng.multivariate_normal(np.zeros(len(cov)), cov, num_samps)

    return samples, cov


def assign_DoF(biadj_mat, deg_of_freedom=None, method="uniform", variances=None):
    """Assign degrees of freedom (latent variables) of VAE to latent factors from causal structure learning
    Parameters
    ----------
    biadj_mat: biadjacency matrix of MCM
    deg_of_freedom: desired size of latent space of VAE
    method: how to distribute excess degrees of freedom to latent causal factors
    variances: diag of covariance matrix over measurement variables

    Returns
    -------
    redundant_biadj_mat: biadjacency matrix specifing VAE structure from latent space to decoder
    """

    num_cliques, num_obs = biadj_mat.shape
    if deg_of_freedom is None:
        # then default to upper bound; TODO: change to max_intersect_num from medil.ecc_algorithms
        deg_of_freedom = num_obs**2 // 4
    elif deg_of_freedom < num_cliques:
        warnings.warn(
            f"Input `deg_of_freedom={deg_of_freedom}` is less than the {num_cliques} required for the estimated causal structure. `deg_of_freedom` increased to {num_cliques} to compensate."
        )
        deg_of_freedom = num_cliques

    if method == "uniform":
        latents_per_clique = np.ones(num_cliques, int) * (deg_of_freedom // num_cliques)
    elif method == "clique_size":
        latents_per_clique = np.round(
            (biadj_mat.sum(1) / biadj_mat.sum()) * (deg_of_freedom - num_cliques)
        ).astype(int)
    elif method == "tot_var" or method == "avg_var":
        clique_variances = biadj_mat @ variances
        if method == "avg_var":
            clique_variances /= biadj_mat.sum(1)
        clique_variances /= clique_variances.sum()
        latents_per_clique = np.round(
            clique_variances * (deg_of_freedom - num_cliques)
        ).astype(int)

    for _ in range(2):
        remainder = deg_of_freedom - latents_per_clique.sum()
        latents_per_clique[np.argsort(latents_per_clique)[0:remainder]] += 1

    redundant_biadj_mat = np.repeat(biadj_mat, latents_per_clique, axis=0)

    return redundant_biadj_mat


class MedilCausalModel(object):
    def __init__(self, biadj_mat=None, latent_dag=None, rng=default_rng(0)):
        self.biadj_mat = biadj_mat
        self.latent_dag = latent_dag
        if biadj_mat is not None:
            self.num_latent, self.num_obs = biadj_mat.shape()
        self.rng = rng
        if biadj_mat is None:
            self.udg = None
        else:
            if latent_dag is None:
                self.udg = biadj_mat.T @ biadj_mat
                np.fill_diagonal(self.udg, False)
            else:
                # take trans closure?
                pass

    def fit(self, dataset, method="lin_gaus"):
        if self.biadj_mat is None:
            # indep test to get U
            # use ECC alg to get biadj_mat
            pass
        pass

    def sample(self, sample_size):
        if hasattr(self, "vae"):
            # samp = sample drawn from vae model
            pass
        else:
            if not hasattr(self, "cov"):
                # generate random weights in +-[0.5, 2]
                num_edges = self.biadj_mat.sum()
                idcs = np.argwhere(self.biadj_mat)
                idcs[:, 1] += self.num_latent

                weights = (self.rng.random(num_edges) * 1.5) + 0.5
                weights[self.rng.choice((True, False), num_edges)] *= -1

                precision = np.eye(self.num_latent + self.num_obs, dtype=float)
                precision[idcs[:, 0], idcs[:, 1]] = weights
                precision = precision.dot(precision.T)

                cov = np.linalg.inv(precision)
                self.cov = cov

            samp = sef.rng.multivariate_normal(
                np.zeros(len(self.cov)), self.cov, sample_size
            )
        return samp

    def rand(self, num_obs, num_latent=None, edge_prob=None):
        self.num_obs = num_obs

        if num_latent is not None and edge_prob is not None:
            raise ValueError(
                "You may specify `num_latent` or `edge_prob` but not both."
            )
        elif num_latent is not None:
            self.num_latent = num_latent
            self.rand_grues()
        else:
            if edge_prob is None:
                edge_prob = 0.5
            self.rand_er(edge_prob)

        return self

    def rand_er(self, edge_prob):
        """Generate minMCM from Erdős–Rényi random undirected graph
        over observed variables."""
        # ER random graph
        udg = np.zeros((self.num_obs, self.num_obs), bool)
        max_edges = (self.num_obs * (self.num_obs - 1)) // 2
        num_edges = np.round(edge_prob * max_edges).astype(int)
        edges = np.ones(max_edges)
        edges[num_edges:] = 0
        udg[np.triu_indices(self.num_obs, k=1)] = self.rng.permutation(edges)
        udg += udg.T
        np.fill_diagonal(udg, True)
        self.udg = udg

        # find latent connections (minimum edge clique cover)
        biadj_mat = find_cm(udg)
        self.biadj_mat = biadj_mat.astype(bool)
        self.num_latent = len(biadj_mat)

    def rand_grues(self):
        """Generate minMCM with specified num_latent by using randomly
        applied Groebner basis moves."""

        # initialize at graph with specified number of latents (so U
        # has L-1 disconnected verts and then a clique of size M-L-1)
        init = np.zeros((self.num_obs, self.num_obs), bool)
        init[self.num_latent - 1 :, :][:, self.num_latent - 1 :] = True
        np.fill_diagonal(init, False)
        reorder = self.rng.permutation(self.num_obs)
        init = init[:, reorder][reorder, :]

        # take random walk from init graph
        dummy_samp = self.rng.random(init.shape)
        rw = rand_walker(dummy_samp, self.rng)
        rw.explore = True
        move_prob = {
            "merge": 0,
            "split": 0,
            "within": 2 / 3,
            "out_del": 1 / 3,
            "out_add": 1 / 3,
        }
        num_moves = 1000
        rw.mcmc(init, move_prob, num_moves)

        self.udg = rw.uec = self.rng.choice(rw.markov_chain)

        # # fing ECC
        # np.fill_diagonal(self.udg, True)
        # self.biadj_mat = find_h(self.udg) # or find_cm for exact

        # # find ECC in polynomial time
        rw.get_max_cpdag()
        max_cpdag = rw.cpdag
        sinks = np.flatnonzero(np.logical_and(max_cpdag.sum(1) == 0, max_cpdag.sum(0)))
        nonsinks = np.delete(np.arange(self.num_obs), sinks)
        order = np.append(nonsinks, sinks)
        dag = np.triu(max_cpdag[:, order][order, :])
        sources = np.flatnonzero(dag.sum(0) == 0)
        dag[sources, sources] = True  # take de(sources) as cliques
        self.biadj_mat = dag[sources, :]
