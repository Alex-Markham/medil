# notes: maybe find_minMCM()
# min='latents' or 'causal_relations' 
# eventually add options: for listing all minMCMs of each type; using quick heuristic alg for just one; and minMCM other than the ones given by the user
from .graph import UndirectedDependenceGraph
import numpy as np


def find_clique_min_cover(graph, verbose=False):
    graph = UndirectedDependenceGraph(graph, verbose)
    try:
        graph.make_aux()
    except ValueError:
        print("The input graph doesn't appear to have any edges!")
        return graph.adj_matrix

    num_cliques = 1
    the_cover = None
    if True:# verbose:
        # find bound for cliques in solution
        max_intersect_num = graph.num_vertices ** 2 // 4
        if max_intersect_num < graph.num_edges:
            p = graph.n_choose_2(graph.num_vertices) - graph.num_edges
            t = int(np.sqrt(p))
            max_intersect_num = p + t if p > 0 else 1
        print("solution has at most {} cliques.".format(max_intersect_num))
    while the_cover is None:
        if True:                # verbose:
            print("\ntesting for solutions with {}/{} cliques".format(num_cliques, max_intersect_num))
        the_cover = branch(graph, num_cliques, the_cover)
        num_cliques += 1

    return the_cover


def branch(graph, k_num_cliques, the_cover):
    branch_graph = graph.reducible_copy()
    # if the_cover is not None:
    #     print(the_cover)
    #     for clique in the_cover:  # this might not be necessary, since the_cover_prime is only +1 clique
    #         print('clique: {}'.format(clique))
    #         branch_graph.the_cover = [clique]
    #         branch_graph.cover_edges()  # only works one clique at a time, or on a list of edges
    branch_graph.the_cover = the_cover
    branch_graph.cover_edges()
    
    if branch_graph.num_edges == 0:
        return branch_graph.reconstruct_cover(the_cover)

    # branch_graph.the_cover = the_cover

    branch_graph.reduzieren(k_num_cliques)
    k_num_cliques = branch_graph.k_num_cliques
    
    if k_num_cliques < 0:
        return None

    if branch_graph.num_edges == 0:  # equiv to len(branch_graph.extant_edges_idx)==0
        return branch_graph.the_cover  # not in paper, but speeds it up slightly; or rather return None?
    
    chosen_nbrhood = branch_graph.choose_nbrhood()
    # print("num cliques: {}".format(len([x for x in max_cliques(chosen_nbrhood)])))
    for clique_nodes in max_cliques(chosen_nbrhood):
        if len(clique_nodes) == 1:  # then this vert has been rmed; quirk of max_cliques
            continue
        clique = np.zeros(branch_graph.unreduced.num_vertices, dtype=int)
        clique[clique_nodes] = 1
        union = clique.reshape(1, -1) if branch_graph.the_cover is None else np.vstack((branch_graph.the_cover, clique))

        the_cover_prime = branch(branch_graph, k_num_cliques-1, union)
        if the_cover_prime is not None:
            return the_cover_prime
    return None


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
