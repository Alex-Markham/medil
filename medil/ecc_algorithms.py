# notes: maybe find_minMCM()
# min='latents' or 'causal_relations' 
# eventually add options: for listing all minMCMs of each type; using quick heuristic alg for just one; and minMCM other than the ones given by the user
from .graph import UndirectedDependenceGraph
import numpy as np


def find_clique_min_cover(graph, verbose=False):
    graph = UndirectedDependenceGraph(graph)
    graph.make_aux()
    # make copies for resetting during loop---could also use deepcopy
    orig_common_neighbors = np.array(graph.common_neighbors)
    orig_nbrhood_edge_counts = np.array(graph.nbrhood_edge_counts)

    counter = 0
    the_cover = None
    if verbose:
        max_intersect_num = graph.num_vertices ** 2 // 4
        print("solution has at most {} cliques.".format(max_intersect_num))
    while the_cover is None:
        if verbose:
            print("\ntesting for solutions with {}/{} cliques".format(counter, max_intersect_num))
        the_cover = branch(graph, counter, the_cover, verbose)
        counter += 1
        # need to reset graph aux back to original now---could also use deepcopy
        graph.common_neighbors = np.array(orig_common_neighbors)
        graph.nbrhood_edge_counts = np.array(orig_nbrhood_edge_counts)
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


def reducee(graph, counter, uncovered_graph, the_cover, verbose):
    # repeatedly apply three reduction rules

    reducing = True
    if verbose:
        print('\t\treducing:')
    while reducing:
        reducing = False
            
        # rule_1: Remove isolated vertices and vertices that are only
        # adjacent to covered edges
        
        # 'remove' (set (i,i) to 0) isolated nodes i (and isolated
        # nodes in uncovered_graph are those adjactent to only covered
        # edges)
        isolated_verts = np.where(uncovered_graph.sum(0)+uncovered_graph.sum(1)==2)[0]
        if len(isolated_verts) > 0: # then Rule 1 is applied
            if verbose:
                print("\t\t\tapplying Rule 1...")

            uncovered_graph[isolated_verts, isolated_verts] = 0

        
        # rule_2: If an uncovered edge {u,v} is contained in exactly one
        # maximal clique C, then add C to the solution, mark its edges as
        # covered, and decrease k by one

        # only check uncovered edges---may cause bugs?
        covered_edges_idx = get_covered_edges_idx(graph, uncovered_graph)

        graph.common_neighbors[covered_edges_idx] = 0  # zeros out a row
        graph.nbrhood_edge_counts[covered_edges_idx] = 0  # zeros out a row

        # edges in at least 1 maximal clique
        at_least = graph.nbrhood_edge_counts > 0

        # edges in at most 1 maximal clique
        at_most = (graph.n_choose_2(graph.common_neighbors.sum(1)) - graph.nbrhood_edge_counts) == 0

        # cliques containing edges in exactly 1 maximal clique
        cliques = graph.common_neighbors[at_least & at_most]

        if cliques.any():       # then apply Rule 2
            if verbose:
                print("\t\t\tapplying Rule 2...")
            cliques = np.unique(cliques, axis=0) # need to fix
            the_cover = cliques if the_cover is None else np.vstack((the_cover, cliques))
            uncovered_graph = cover_edges(uncovered_graph, cliques, verbose)
            counter -= 1
            continue            # start the reducee loop over so Rule
                                # 1 can 'clean up'

        
        # rule_3: Consider a vertex v that has at least one
        # guest. If inhabitants (of the neighborhood) occupy the
        # gates, then delete v. To reconstruct a solution for the
        # unreduced instance, add v to every clique containing a
        # guest of v. (prisoneers -> guests; dominate ->
        # occupy; exits -> gates; hierarchical social structures and
        # relations needn't be reproduced in mathematical
        # abstractions)

        exits = np.zeros((uncovered_graph.shape), dtype=bool)
            
        for vert, nbrhood in enumerate(uncovered_graph):
            if nbrhood[vert]==0: # then nbrhood is empty
                continue
            nbrs = np.flatnonzero(nbrhood)
            for nbr in nbrs:
                if (nbrhood - uncovered_graph[nbr] == -1).any():
                    exits[vert, nbr] = True
        # exits[i, j] == True iff j is an exit for i

        # guests[i, j] == True iff j is a guest of i
        guests = np.logical_and(~exits, uncovered_graph)

        applied_3 = False
        for pair in np.transpose(np.where(guests)):
            if the_cover is None:
                break
            guest_rooms_idx = np.transpose(np.where(the_cover[:, pair[1]]))
            
            if np.logical_not(the_cover[guest_rooms_idx, pair[1]]).any(): # then apply rule
                applied_3 = True
                if verbose:
                    print("\t\t\tapplying Rule 3...")
            # add host to all cliques containing guest
            the_cover[guest_rooms_idx, pair[1]] = 1
            uncovered_graph = cover_edges(uncovered_graph, the_cover, verbose)
        if applied_3:
            continue
            
    return graph, counter, uncovered_graph, the_cover


def cover_edges(graph_adj_mat, the_cover, verbose):
    if the_cover is None:
        return graph_adj_mat
    
    # slightly more convenient representation and obviates need for deepcopy
    uncovered_graph = np.triu(graph_adj_mat)
    
    # change edges to 0 if they're covered
    for clique in the_cover:
        covered = clique.nonzero()[0]
        
        # trick for getting combinations from idx
        comb_idx = np.triu_indices(len(covered), 1)
        
        # actual combinations
        covered_row = covered[comb_idx[0]]
        covered_col = covered[comb_idx[1]]
        
        # cover edges
        uncovered_graph[covered_row, covered_col] = 0

    if verbose:
        print("\t\t\t{} uncovered edges remaining".format(uncovered_graph.sum()))
    return np.triu(uncovered_graph, 1) + uncovered_graph.T


def get_covered_edges_idx(graph, uncovered_graph):
    return np.array([graph.get_idx(x) for x in np.transpose(np.where(np.logical_and(uncovered_graph==0, np.tri(graph.num_vertices, k=-1).T)))], dtype=int) # grooosssssssssssssss



def choose_edge(graph):    
    score = graph.n_choose_2(graph.common_neighbors.sum(1)) - graph.nbrhood_edge_counts
    # score includes scores for non-existent edges, so have exclude those, otherwise could use .argmin()
    chosen_edge_idx = np.where(score==score[graph.extant_edges_idx].min())[0][0]
    return chosen_edge_idx


def max_cliques(nbrhood):
    # pieced together from nx.from_numpy_array and nx.find_cliques
    # nx.find_cliques is output sensitive :)

    # convert adjacency matrix to nx style graph
    
    # adapted from nx.find_cliques to find max cliques
    if len(nbrhood) == 0:
        return

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
