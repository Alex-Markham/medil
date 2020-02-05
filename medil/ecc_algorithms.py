# notes: maybe find_minMCM()
# min='latents' or 'causal_relations' 
# eventually add options: for listing all minMCMs of each type; using quick heuristic alg for just one; and minMCM other than the ones given by the user
from .graph import UndirectedDependenceGraph
import numpy as np


def find_clique_min_cover(graph, verbose=False):
    graph = UndirectedDependenceGraph(graph)
    graph.make_aux()

    num_cliques = 0
    the_cover = None
    if verbose:
        max_intersect_num = graph.num_vertices ** 2 // 4
        print("solution has at most {} cliques.".format(max_intersect_num))
    while the_cover is None:
        if verbose:
            print("\ntesting for solutions with {}/{} cliques".format(num_cliques, max_intersect_num))
        the_cover = branch(graph, num_cliques, the_cover, verbose)
        num_cliques += 1
    return the_cover


def branch(graph, counter, the_cover, verbose):
    uncovered_graph = cover_edges(graph.adj_matrix, the_cover, verbose)
    if not np.any(uncovered_graph):
        return the_cover

    if verbose:
        print("\tbranching...")
    reduction = reducee(graph, counter, uncovered_graph, the_cover, verbose)  # reduced = (reduciable)graph.reduce(...) and inpit is graph.reducable() and then can reset in main loop in finde_cover
    # now graph_aux is the uncovered_graph, not the original---graph.common_neighbors and graph.nbrhood_edge_counts ??
    graph, counter, uncovered_graph, the_cover = reduction

    if counter < 0:
        return None

    chosen_edge = choose_edge(graph)
    chosen_nbrhood = graph.nbrhood(chosen_edge)
    for clique_nodes in max_cliques(chosen_nbrhood):
        clique = np.zeros(graph.num_vertices, dtype=int)
        clique[clique_nodes] = 1
        union = clique.reshape(1, -1) if the_cover is None else np.vstack((the_cover, clique))
        the_cover_prime = branch(graph, counter-1, union, verbose)
        if the_cover_prime is not None:
            return the_cover_prime
    return the_cover
