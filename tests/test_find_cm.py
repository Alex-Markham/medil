import numpy as np
from medil.ecc_algorithms import find_clique_min_cover as find_cm
from medil.graph import UndirectedDependenceGraph
from medil.ecc_algorithms import branch
from medil.ecc_algorithms import max_cliques

# Here are some integration tests.
# def test_find_cm_on_trivial_edgeless_graph():
#     # not so important at the moment
#     graph = np.array([[]], int)
#     cover = find_cm(graph, True)

#     graph = np.zeros((5, 5), int)
#     cover = find_cm(graph, True)


def test_find_cm_on_3_cycle():
    cycle_3 = np.ones((3, 3), dtype=int)
    cover = find_cm(cycle_3, True)
    assert cover.shape==(1, 3)
    assert ~np.any(cover - [1, 1, 1], axis=1)


def test_reduction_rule_1_on_3cycle_plus_isolated():
    graph = np.zeros((4, 4), dtype=int)  # init
    graph[1:4, 1:4] = 1         # add 3cycle
    graph[0, 0] = 1             # add isolated vert

    cover = find_cm(graph, verbose=True)
    
    assert cover.shape==(1, 4)
    assert ~np.any(cover - [0, 1, 1, 1], axis=1)


def test_find_cm_on_triangle():
    graph_triangle = np.asarray([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1]])

    cover = find_cm(graph_triangle, True)
    assert cover.shape==(3, 6)

    correct_cover = np.array([[0, 0, 1, 0, 1, 1],
                              [0, 1, 0, 1, 1, 0],
                              [1, 1, 1, 0, 0, 0]])
    
    assert (~np.any(cover - correct_cover[0], axis=1)).any()
    assert (~np.any(cover - correct_cover[1], axis=1)).any()
    assert (~np.any(cover - correct_cover[2], axis=1)).any()
    

def test_find_cm_on_clean_am_cm_diff():
    graph = np.array([[1, 1, 1, 1, 1, 0, 1, 0],
                         [1, 1, 1, 0, 1, 1, 1, 1],
                         [1, 1, 1, 0, 1, 1, 1, 1],
                         [1, 0, 0, 1, 0, 1, 1, 1], 
                         [1, 1, 1, 0, 1, 1, 0, 1],
                         [0, 1, 1, 1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 0, 0, 1, 1],
                         [0, 1, 1, 1, 1, 1, 1, 1]])
     
    cover = find_cm(graph, verbose=True)
    assert cover.shape==(5, 8)
    correct_cover = np.array([[1, 1, 1, 0, 1, 0, 0, 0],
                              [1, 0, 0, 1, 0, 0, 1, 0],
                              [0, 1, 1, 0, 1, 1, 0, 1],
                              [0, 0, 0, 1, 0, 1, 0, 1],
                              [0, 1, 1, 0, 0, 0, 1, 1]])


    assert (~np.any(cover - correct_cover[0], axis=1)).any()
    assert (~np.any(cover - correct_cover[1], axis=1)).any()
    assert (~np.any(cover - correct_cover[2], axis=1)).any()
    assert (~np.any(cover - correct_cover[3], axis=1)).any()
    assert (~np.any(cover - correct_cover[4], axis=1)).any()  # not from rule 2


# def test_real_data():
#     results = np.load("/home/alex/Projects/mcm_paper/uai_2020/data_analysis/monte_carlo_test_results_1000.npz")
#     all_deps = results['deps']

#     deps = all_deps[2:63, 2:63]

#     cover = find_cm(deps.astype(int), verbose=True)
#     print(cover)
# test_real_data()

# Here are unit tests.

def test_make_aux_on_triangle():
    graph_triangle = np.asarray([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1]])
    graph = UndirectedDependenceGraph(graph_triangle)
    graph.make_aux()

    # non-edges:
    assert (graph.common_neighbors[[2, 3, 4, 8, 9, 13]] == 0).all()
    # edges:
    assert (graph.common_neighbors[[0, 1]] == [1, 1, 1, 0, 0, 0]).all()
    assert (graph.common_neighbors[5]      == [1, 1, 1, 0, 1, 0]).all()
    assert (graph.common_neighbors[6]      == [0, 1, 0, 1, 1, 0]).all()
    assert (graph.common_neighbors[7]      == [0, 1, 1, 1, 1, 0]).all()
    assert (graph.common_neighbors[10]     == [0, 1, 1, 0, 1, 1]).all()
    assert (graph.common_neighbors[11]     == [0, 0, 1, 0, 1, 1]).all()
    assert (graph.common_neighbors[12]     == [0, 1, 0, 1, 1, 0]).all()
    assert (graph.common_neighbors[14]     == [0, 0, 1, 0, 1, 1]).all()

    assert (graph.nbrhood_edge_counts == [3, 3, 0, 0, 0, 5, 3, 5, 0, 0, 5, 3, 3, 0, 3]).all()

    
def test_reduce_rule_1_on_isolated_plus_3cycle():
    graph = np.zeros((4, 4), dtype=int)  # init
    graph[1:4, 1:4] = 1         # add 3cycle


    # correct answer (assuming make_aux is correct)
    correct = UndirectedDependenceGraph(graph)
    correct.make_aux()

    # to test
    graph[0, 0] = 1             # add isolated vert
    being_tested = UndirectedDependenceGraph(graph).reducible_copy()
    being_tested.rule_1()
    
    assert (correct.num_vertices == being_tested.num_vertices).all()
    assert (correct.common_neighbors == being_tested.common_neighbors).all()
    assert (correct.nbrhood_edge_counts == being_tested.nbrhood_edge_counts).all()    


def test_find_max_cliques():
    graph = np.array([[1, 1, 1, 1, 1, 0, 1, 0],
                      [1, 1, 1, 0, 1, 1, 1, 1],
                      [1, 1, 1, 0, 1, 1, 1, 1],
                      [1, 0, 0, 1, 0, 1, 1, 1], 
                      [1, 1, 1, 0, 1, 1, 0, 1],
                      [0, 1, 1, 1, 1, 1, 0, 1],
                      [1, 1, 1, 1, 0, 0, 1, 1],
                      [0, 1, 1, 1, 1, 1, 1, 1]])

    graph = UndirectedDependenceGraph(graph).reducible_copy()
    graph.k_num_cliques = 5
    graph.rule_2()
    graph.rule_2()
    graph.rule_1()    
    graph.rule_2()
    graph.rule_1()    
    graph.rule_2()
    graph.rule_1()
 
    
def test_reduce_rule_1_on_triangle():
    graph_triangle = np.asarray([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1]])

    
def test_reduce_rule_2_3cycle():
    init = np.ones((3, 3), dtype=int)
    
    graph = UndirectedDependenceGraph(init).reducible_copy()
    graph.k_num_cliques = 1
    graph.rule_2()

    assert graph.k_num_cliques == 0
    assert graph.the_cover.shape == (1, 3)
    assert ([1, 1, 1]==graph.the_cover).all()


def test_cover_edges():
    graph_triangle = np.asarray([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 1]])

    clique = np.array([1, 1, 1, 0, 0, 0], int)

    being_tested = UndirectedDependenceGraph(graph_triangle).reducible_copy()
    being_tested.the_cover = clique.reshape(1, 6)
    being_tested.cover_edges()

    assert (being_tested.adj_matrix[0] == [1, 0, 0, 0, 0, 0]).all()


def test_reduce_rule_3_real_data():
    results = np.load("/home/alex/Projects/mcm_paper/uai_2020/data_analysis/monte_carlo_test_results_1000.npz")
    all_deps = results['deps']

    deps = all_deps[2:63, 2:63]

    c0_idx = [2, 3, 15, 17, 19, 29, 33, 39, 49, 52, 54, 55]
    c0_deps = deps[:, c0_idx][c0_idx, :]

    graph = UndirectedDependenceGraph(np.array(c0_deps, int)).reducible_copy()
        
    
# def test_find_cm_on_2clique_house():
#     grapyh = 5 nodes, 1 4-clique sharing an edge with a 3 clique



def full_test():
    results = np.load("/home/alex/Projects/mcm_paper/uai_2020/data_analysis/monte_carlo_test_results_1000.npz")
    all_deps = results['deps']

    deps = all_deps[2:63, 2:63]

    c0_idx = [2, 3, 15, 17, 19, 29, 33, 39, 49, 52, 54, 55]
    c0_deps = deps[:, c0_idx][c0_idx, :]

    graph = UndirectedDependenceGraph(np.array(c0_deps, int)).reducible_copy()

    edges = [graph.get_edge(edge_idx) for edge_idx in graph.extant_edges_idx]  # same as zipped np.nonzero(np.triu(graph.adj_matrix, 1))

    nbrss = np.array([graph.nbrs(edge) for edge in edges])
    
    score = self.n_choose_2(self.common_neighbors.sum(1)) - self.nbrhood_edge_counts


def test_find_cm_on_clean_am_cm_diff():
    graph = np.array([[1, 1, 1, 1, 1, 0, 1, 0],
                      [1, 1, 1, 0, 1, 1, 1, 1],
                      [1, 1, 1, 0, 1, 1, 1, 1],
                      [1, 0, 0, 1, 0, 1, 1, 1], 
                      [1, 1, 1, 0, 1, 1, 0, 1],
                      [0, 1, 1, 1, 1, 1, 0, 1],
                      [1, 1, 1, 1, 0, 0, 1, 1],
                      [0, 1, 1, 1, 1, 1, 1, 1]])

    graph = UndirectedDependenceGraph(graph).reducible_copy()

    score = graph.n_choose_2(graph.common_neighbors.sum(1)) - graph.nbrhood_edge_counts

    hmmm = np.where(score[graph.extant_edges_idx]==0)[0]

    max_cliques_in_solution = graph.common_neighbors[graph.extant_edges_idx[hmmm]]

    solution = np.unique(max_cliques_in_solution, axis=0)
