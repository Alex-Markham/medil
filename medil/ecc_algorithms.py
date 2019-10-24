# find_minMCM()
# min='latents' or 'causal_relations' 
# eventually add options: for listing all minMCMs of each type; using quick heuristic alg for just one; and minMCM other than the ones given by the user

def find_clique_min_cover(graph):
    # graph should be UndirectedDependenceGraph object
    graph.make_aux()

    counter = 0
    the_cover = None
    while the_cover is None:
        the_cover = branch(graph, counter, the_cover)
        counter += 1
    return the_cover


def branch(graph, counter, the_cover):
    pass
