import numpy as np

# from .grues import InputData


# class MicroModel(object):
#     def __init__(self, num_nodes, edge_density, rng=np.random.default_rng(0)):
#         self.num_nodes = num_nodes
#         self.rng = rng

#         # generate DAG
#         self.dag = np.zeros((num_nodes, num_nodes), bool)
#         max_edges = (num_nodes * (num_nodes - 1)) // 2
#         self.dag[np.triu_indices(num_nodes, k=1)] = rng.choice(
#             a=(True, False), size=max_edges, p=(edge_density, 1 - edge_density)
#         )

#         # generate covariance matrix
#         num_edges = self.dag.sum()

#         weights = (rng.random(num_edges) * 1.5) + 0.5
#         weights[rng.choice((True, False), num_edges)] *= -1

#         concentration = (np.eye(num_nodes) + self.dag).astype(float)
#         concentration[self.dag] = -weights

#         self.covariance = np.linalg.inv(concentration @ concentration.T)

#         # compute macromodel
#         trans_closure = np.linalg.matrix_power(
#             np.eye(num_nodes, dtype=bool) + self.dag, num_nodes - 1
#         )
#         self.uec = (trans_closure.T @ trans_closure).astype(bool)

#         helper = InputData.__new__(InputData)
#         helper.uec, helper.num_feats = self.uec, num_nodes
#         helper.get_max_cpdag()
#         helper.reduce_max_cpdag()

#         self.macro_dag = helper.dag_reduction
#         avg_const = 1 / helper.chain_comps.sum(1)[:, None]
#         self.macro_covariance = (
#             (avg_const * helper.chain_comps) @ self.covariance @ helper.chain_comps.T
#         )

#     def sample(self, samp_size):
#         mu = np.zeros(self.num_nodes)
#         return self.rng.multivariate_normal(mu, self.covariance, samp_size)
