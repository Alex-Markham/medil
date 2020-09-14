import numpy as np
from medil.graph import UndirectedDependenceGraph
import medil.examples as ex


def test_make_aux_on_triangle():
    graph = UndirectedDependenceGraph(ex.triangle.UDG)
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


def test_reduce_rule_2_3cycle():
    init = np.ones((3, 3), dtype=int)
    
    graph = UndirectedDependenceGraph(init).reducible_copy()
    graph.k_num_cliques = 1
    graph.rule_2()

    assert graph.k_num_cliques == 0
    assert graph.the_cover.shape == (1, 3)
    assert ([1, 1, 1]==graph.the_cover).all()


def test_cover_edges():
    triangle_udg = ex.triangle.UDG

    clique = np.array([1, 1, 1, 0, 0, 0], int)

    being_tested = UndirectedDependenceGraph(triangle_udg).reducible_copy()
    being_tested.the_cover = clique.reshape(1, 6)
    being_tested.cover_edges()

    assert (being_tested.adj_matrix[0] == [1, 0, 0, 0, 0, 0]).all()


def test_reduce_rule_3_example_from_paper():
    graph = np.array([[1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                      [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]], int)

    graph = UndirectedDependenceGraph(np.array(graph)).reducible_copy()

    # graph.verbose = True
    graph.rule_3()

    # correct info stored for reconstruction
    assert graph.reduced_away.sum(1)[4]

    # adj matrix updated correctly
    assert graph.adj_matrix.sum(0)[4] == 1
    assert graph.adj_matrix.sum(1)[4] == 1

    
def test_reconstruct_rule_3_example_from_paper():
    graph = np.array([[1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                      [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]], int)

    graph = UndirectedDependenceGraph(np.array(graph)).reducible_copy()

    # graph.verbose = True
    graph.rule_3()
    graph.reduzieren(7)

    # reduced_away vert isn't in cover
    assert graph.the_cover.sum(0)[4]==0

    cover = graph.the_cover
    # correct cover otherwise
    correct_reduced_cover = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    assert (~np.any(cover - correct_reduced_cover[0], axis=1)).any()
    assert (~np.any(cover - correct_reduced_cover[1], axis=1)).any()
    assert (~np.any(cover - correct_reduced_cover[2], axis=1)).any()
    assert (~np.any(cover - correct_reduced_cover[3], axis=1)).any()
    assert (~np.any(cover - correct_reduced_cover[4], axis=1)).any()
    assert (~np.any(cover - correct_reduced_cover[5], axis=1)).any()
    assert (~np.any(cover - correct_reduced_cover[6], axis=1)).any()

    cover = graph.reconstruct_cover(graph.the_cover)
    correct_recon_cover = np.array([[0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
                                    [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    
    assert (~np.any(cover - correct_recon_cover[0], axis=1)).any()
    assert (~np.any(cover - correct_recon_cover[1], axis=1)).any()
    assert (~np.any(cover - correct_recon_cover[2], axis=1)).any()
    assert (~np.any(cover - correct_reduced_cover[0], axis=1)).any()
    assert (~np.any(cover - correct_reduced_cover[1], axis=1)).any()
    assert (~np.any(cover - correct_reduced_cover[4], axis=1)).any()
    assert (~np.any(cover - correct_reduced_cover[5], axis=1)).any()
    
    
# def test_reduce_rule_3_real_data():
    # results = np.load("/home/alex/Projects/mcm_paper/uai_2020/data_analysis/monte_carlo_test_results_1000.npz")
    # all_deps = results['deps']

    # deps = all_deps[2:63, 2:63]

    # c0_idx = [2, 3, 15, 17, 19, 29, 33, 39, 49, 52, 54, 55]
    # c0_deps = deps[:, c0_idx][c0_idx, :]

    # graph = UndirectedDependenceGraph(np.array(c0_deps, int)).reducible_copy()
