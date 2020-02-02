def reduction(graph, counter, uncovered_graph, the_cover, verbose):
    # repeatedly apply three reduction rules

    reducing = True
    if verbose:
        print('\t\treducing:')
    while reducing:
        reducing = False
        pass

    return graph, counter, uncovered_graph, the_cover


def apply_rule_1(uncovered_graph, verbose):
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

    return uncovered_graph


def apply_rule_2(graph, counter, uncovered_graph, the_cover, ):
    # rule_2: If an uncovered edge {u,v} is contained in exactly one
    # maximal clique C, then add C to the solution, mark its edges as
    # covered, and decrease k by one
    
    # only check uncovered edges---may cause bugs?
    covered_edges_idx = get_covered_edges_idx(graph, uncovered_graph)

    # TRY removing groph., so it's just a copy
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

    return


def apply_rule_3():
    return








            
        

        
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
            

