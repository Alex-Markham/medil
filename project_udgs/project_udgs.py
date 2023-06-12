import numpy as np

from gues.grues import InputData as rand_walker

rng = np.random.default_rng(0)

mnist_udg = np.load("project_udgs/mnist_udg-0.001.npy")
tcga_udg = np.load("project_udgs/tcga_udg-0.05.npy")
tumor_udg = np.load("project_udgs/tumor_udg-0.05.npy")


def rm_project(udg):
    num_obs = len(udg)
    np.fill_diagonal(udg, False)

    dummy_samp = rng.random(udg.shape)
    rw = rand_walker(dummy_samp, rng)

    # # find ECC in polynomial time
    rw.init_uec(udg)
    rw.get_max_cpdag()
    max_cpdag = rw.cpdag
    sinks = np.flatnonzero(np.logical_and(max_cpdag.sum(1) == 0, max_cpdag.sum(0)))
    nonsinks = np.delete(np.arange(num_obs), sinks)
    order = np.append(nonsinks, sinks)
    dag = np.triu(max_cpdag[:, order][order, :])
    sources = np.flatnonzero(dag.sum(0) == 0)
    dag[sources, sources] = True  # take de(sources) as cliques
    biadj_mat = dag[sources, :]
    return biadj_mat


# mnist_rm_projected = rm_project(mnist_udg)
# tcga_rm_projected = rm_project(tcga_udg)
# tumor_rm_projected = rm_project(tumor_udg)

# np.save("project_udgs/rm_mnist_projected.npy", rm_mnist_projected)
# np.save("project_udgs/rm_tcga_projected.npy", rm_tcga_projected)
# np.save("project_udgs/rm_tumor_projected.npy", rm_tumor_projected)

# rm_mnist_projected = np.load("project_udgs/rm_mnist_projected.npy")
# rm_tcga_projected = np.load("project_udgs/rm_tcga_projected.npy")
# rm_tumor_projected = np.load("project_udgs/rm_tumor_projected.npy")


def add_project(udg):
    num_obs = len(udg)
    np.fill_diagonal(udg, False)

    U = np.copy(udg)
    compliment_U = ~U
    np.fill_diagonal(compliment_U, False)

    # V_ij == 1 if and only if there's a k adjacent to j but not i
    V = compliment_U @ U

    # W_ij == 1 if and only if there's k such that i--j--k is an induced path
    W = np.logical_and(V, U).T

    # This orients all v-structures and removes edges violating CI relations
    U[W] = False
