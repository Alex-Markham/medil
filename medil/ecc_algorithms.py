# notes: maybe find_minMCM()
# min='latents' or 'causal_relations' 
# eventually add options: for listing all minMCMs of each type; using quick heuristic alg for just one; and minMCM other than the ones given by the user
from .graph import UndirectedDependenceGraph
import numpy as np


def find_clique_min_cover(graph, verbose=False):
    graph = UndirectedDependenceGraph(graph, verbose)
    graph.make_aux()

    num_cliques = 0
    the_cover = None
    if verbose:
        max_intersect_num = graph.num_vertices ** 2 // 4
        print("solution has at most {} cliques.".format(max_intersect_num))
    while the_cover is None:
        if verbose:
            print("\ntesting for solutions with {}/{} cliques".format(num_cliques, max_intersect_num))
        reducible_graph = graph.reducible_copy()
        the_cover = branch(reducible_graph, num_cliques, the_cover)
        assert num_cliques < max_intersect_num
        num_cliques += 1
    return the_cover


def branch(reducible_graph, k_num_cliques, the_cover):
    reducible_graph.the_cover = the_cover
    reducible_graph.cover_edges()
    if reducible_graph.num_edges == 0:
        return the_cover

    if reducible_graph.verbose:
        print("\tbranching...")

    reducible_graph.reduzieren(k_num_cliques)
    k_num_cliques = reducible_graph.k_num_cliques
    
    if k_num_cliques < 0:
        return None

    chosen_nbrhood = reducible_graph.choose_nbrhood()
    for clique_nodes in max_cliques(chosen_nbrhood):
        clique = np.zeros(reducible_graph.unreduced.num_vertices, dtype=int)
        clique[clique_nodes] = 1
        union = clique.reshape(1, -1) if reducible_graph.the_cover is None else np.vstack((reducible_graph.the_cover, clique))
        the_cover_prime = branch(reducible_graph, k_num_cliques-1, union)
        if the_cover_prime is not None:
            return the_cover_prime
    return reducible_graph.the_cover


def max_cliques(nbrhood):
    # pieced together from nx.from_numpy_array and nx.find_cliques,
    # which is output sensitive :)
    
    if len(nbrhood) == 0:
        return

    # convert adjacency matrix to nx style graph
    # adapted from nx.find_cliques to find max cliques

    adj = {u: {v for v in np.nonzero(nbrhood[u])[0] if v != u} for u in range(len(nbrhood))}
    Q = [None]

    subg = set(range(len(nbrhood)))
    cand = set(range(len(nbrhood)))
    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]
    stack = []

    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass    
    # note: max_cliques is a generator, so it's consumed after being
    # looped through once
