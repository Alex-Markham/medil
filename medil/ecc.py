import numpy as np
from networkx import find_cliques, from_numpy_array
import numba


def cm_cover(graph):
    # an n-vertice graph is represented by an n-by-n adjacency np
    # 2d-array

    
    # initialize auxiliary data structures to make the many
    # applications of Rule 2 (in reducee(), in branch()) more
    # efficient
    graph_aux = make_aux(graph)

    # find the_cover by recursively calling branch() using the
    # parameter counter
    counter = 0
    the_cover = None
    while the_cover is None:
        the_cover = branch(graph, graph_aux, counter, the_cover)
        counter += 1
    return the_cover


def make_aux(graph):
    num_vertices = graph.shape[0]
    num_edges = n_choose_2(num_vertices)
    triu_idx = np.triu_indices(num_vertices, 1)
    
    # find neighbourhood for each vertex
    # each row corresponds to a unique edge
    common_neighbors= np.zeros((num_edges, num_vertices), int) # init
  

    # mapping of edges to unique row idx
    nghbrhd_idx = np.zeros((num_vertices, num_vertices), int)
    nghbrhd_idx[triu_idx] = np.arange(num_edges)
    # nghbrhd_idx += nghbrhd_idx.T
    get_idx = lambda edge: nghbrhd_idx[edge[0], edge[1]]
    
    # reverse mapping
    # edges_idx = np.transpose(triu_idx)
    
    # compute actual neighborhood for each edge = (v_1, v_2)
    nbrs = lambda edge: np.logical_and(graph[edge[0]], graph[edge[1]])    

    extant_edges = np.transpose(np.triu(graph, 1).nonzero())
    extant_nbrs = np.array([nbrs(edge) for edge in extant_edges])
    extant_nbrs_idx = [get_idx(edge) for edge in extant_edges]
    
    common_neighbors[extant_nbrs_idx] = extant_nbrs

    # number of cliques for each node? if we set diag=0
    # num_cliques = common_neighbors.sum(0)

    # sum() of submatrix of graph containing exactly the rows/columns
    # corresponding to the nodes in common_neighbors(edge) using
    # logical indexing:

    # make mask to identify subgraph (closed common neighborhood of
    # nodes u, v in edge u,v)
    mask = lambda edge_idx: np.array(common_neighbors[edge_idx], dtype=bool)

    # make subgraph-adjacency matrix, and then subtract diag and
    # divide by two to get num edges in subgraph---same as sum() of
    # triu(subgraph-adjacency matrix) but probably a bit faster
    nbrhood = lambda edge_idx: graph[mask(edge_idx)][:, mask(edge_idx)]
    num_edges_in_nbrhood = lambda edge_idx: (nbrhood(edge_idx).sum() - mask(edge_idx).sum()) // 2

    nbrhood_edge_counts = np.array([num_edges_in_nbrhood(edge_idx) for edge_idx in np.arange(num_edges)])
    
    return common_neighbors, nbrhood_edge_counts, nbrhood, get_idx


def branch(graph, graph_aux, k, the_cover):
    uncovered_graph = cover_edges(graph, the_cover)
    if (uncovered_graph == 0).all():
        return the_cover

    reduction = reducee(graph, graph_aux, k, uncovered_graph, the_cover)
    # now graph_aux is the uncovored_graph, not the original
    graph, graph_aux, k, uncovered_graph, the_cover = reduction

    if k < 0:
        return None

    chosen_edge = choose_edge(graph_aux)
    nbrhood = graph_aux[2]
    chosen_nbrhood = nbrhood(chosen_edge)
    for clique_nodes in max_cliques(chosen_nbrhood):
        clique = np.zeros(len(graph), dtype=int)
        clique[clique_nodes] = 1

        union = clique if the_cover is None else np.vstack((the_cover, clique))
        the_cover_prime = branch(graph, graph_aux, k-1, union)
        if the_cover_prime is not None:
            return the_cover_prime
    return the_cover


def reducee(graph, graph_aux, k, uncovered_graph, the_cover):
    # repeatedly apply three reduction rules

    common_neighbors, nbrhood_edge_counts, nbrhood, get_idx = graph_aux

    reducing = True

    while reducing:
        reducing = False
            
        # rule_1: Remove isolated vertices and vertices that are only
        # adjacent to covered edges

        # 'remove' (set (i,i) to 0) isolated nodes i (and isolated
        # nodes in uncovered_graph are those adjactent to only covered
        # edges)
        isolated_verts = np.where(uncovered_graph.sum(0)+uncovered_graph.sum(1)==2)[0]
        if len(isolated_verts) > 0: # then Rule 1 was applied
            uncovered_graph[isolated_verts, isolated_verts] = 0


        # rule_2: If an uncovered edge {u,v} is contained in exactly one
        # maximal clique C, then add C to the solution, mark its edges as
        # covered, and decrease k by one

        # only check uncovered edges---may cause bugs?
        covered_edges_idx = np.array([get_idx(x) for x in np.transpose(np.where(np.logical_and(uncovered_graph==0, np.tri(test_graph.shape[0], k=-1).T)))]) # grooosssssssssssssss

        common_neighbors[covered_edges_idx] = 0
        nbrhood_edge_counts[covered_edges_idx] = 0

        graph_aux = common_neighbors, nbrhood_edge_counts, graph_aux[2], graph_aux[3]
        
        # edges in at least 1 maximal clique
        at_least = nbrhood_edge_counts > 0

        # edges in at most 1 maximal clique
        at_most = (n_choose_2(common_neighbors.sum(1)) - nbrhood_edge_counts) == 0

        # cliques containing edges in exactly 1 maximal clique
        cliques = common_neighbors[at_least & at_most]

        if cliques.any():       # then apply Rule 2
            cliques = np.unique(cliques, axis=0) # need to fix
            the_cover = cliques if the_cover is None else np.vstack((the_cover, cliques))
            uncovered_graph = cover_edges(uncovered_graph, cliques)
            k -= 1
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
            guest_rooms_idx = np.tranpose(np.where(the_cover[:, pair[1]]))
            
            if np.logical_not(the_cover[guest_rooms_idx, pair[1]]).any(): # then apply rule
                applied_2 = True
                # add host to all cliques containing guest
                the_cover[guest_rooms_idx, pair[1]] = 1
                uncovered_graph = cover_edges(uncovered_graph, the_cover)
        if applied_3:
            continue
            
    return graph, graph_aux, k, uncovered_graph, the_cover


def cover_edges(graph, the_cover):
    if the_cover is None:
        return graph
    
    # slightly more convenient representation and obviates need for deepcopy
    uncovered_graph = np.triu(graph)
    
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

    return np.triu(uncovered_graph, 1) + uncovered_graph.T
    

def choose_edge(graph_aux):
    common_neighbors, nbrhood_edge_counts, nbrhoods, get_idx = graph_aux
    
    score = n_choose_2(common_neighbors.sum(1)) - nbrhood_edge_counts

    chosen_edge_idx = np.where(score==score.min())[0][0]
    return chosen_edge_idx


def max_cliques(nbrhood):
    # wrapper for nx.find_cliques
    # nx.find_cliques is output sensitive :)

    # convert adjacency matrix to nx graph
    subgraph = from_numpy_array(nbrhood)
    
    # use nx to find max cliques
    max_cliques = find_cliques(subgraph)
    
    # note: max_cliques is a generator, so it's consumed after being
    # looped through once
    return max_cliques


def n_choose_2(n):
    return n * (n - 1) // 2


# def get_idx(edge, graph):
#     num_vertices = graph.shape[0]
#     num_edges = n_choose_2(num_vertices)
#     triu_idx = np.triu_indices(num_vertices, 1)

#     # mapping of edges to unique row idx
#     nghbrhd_idx = np.zeros((num_vertices, num_vertices), int)
#     nghbrhd_idx[triu_idx] = np.arange(num_edges)
#     # nghbrhd_idx += nghbrhd_idx.T

#     return get_idx = lamda edge: nghbrhd_idx[edge[0], edge[1]


# def am_cover(x):
#     pass

# cm and am heuristics?

# optimize for causality/MLMs, maybe making it score-based instead of
# constraint, so that it adds/removes (un)likely edges (i.e., assume
# high clustering co-efficient)

########################################################################################
# test

start = time.time()
for _ in range(10000):
    ecc = cm_cover(test_graph)
end = time.time()
print((end - start) / 10000)
