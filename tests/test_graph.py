import numpy as np
from medil.graph import UndirectedDependenceGraph
from medil.ecc_algorithms import branch
from medil.ecc_algorithms import max_cliques


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
    graph.verbose = True
    graph.rule_2()
    # print(graph.adj_matrix)
    graph.rule_2()
    # print(graph.adj_matrix)
    graph.rule_1()
    # print(graph.adj_matrix)
    graph.rule_2()
    # print(graph.adj_matrix)
    graph.rule_1()    
    # print(graph.adj_matrix)
    graph.rule_2()
    # print(graph.adj_matrix)
    graph.rule_1()
    # print(graph.adj_matrix)
    
    score = graph.n_choose_2(graph.common_neighbors.sum(1)) - graph.nbrhood_edge_counts


            # max_num_edges = self.n_choose_2(self.unreduced.max_num_verts)
        # mask = lambda edge_idx: np.array(self.common_neighbors[edge_idx], dtype=bool)
        
        # # make subgraph-adjacency matrix, and then subtract diag and
        # # divide by two to get num edges in subgraph---same as sum() of
        # # triu(subgraph-adjacency matrix) but probably a bit faster
        # nbrhood = lambda edge_idx: self.adj_matrix[mask(edge_idx)][:, mask(edge_idx)]
        # max_num_edges_in_nbrhood = lambda edge_idx: (nbrhood(edge_idx).sum() - mask(edge_idx).sum()) // 2

        # # from paper: set of c_{u, v} for all edges (u, v)
        # self.nbrhood_edge_counts = np.array([max_num_edges_in_nbrhood(edge_idx) for edge_idx in np.arange(max_num_edges)], int)
        # # assert (nbrhood_edge_counts==self.nbrhood_edge_counts).all()
        # # print(nbrhood_edge_counts, self.nbrhood_edge_counts)
        # # need to fix!!!!!!!! update isn't working; so just recomputing for now
        # # # # # # # # but actually update produces correct result though recomputing doesn't?
