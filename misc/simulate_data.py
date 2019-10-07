import numpy as np


def simulate_data(arrray):
    # this will be the function to be called; it will take a
    # num_main_graphs X num_erroneous_graphs X 2 array; each element
    # (i, 0, 0) in the first column of the XY-plane specifies the
    # number of nodes in a graph (main_graph_i); the values of the jth
    # element in the remaining columns (i, j, 0) specify the number of
    # edges to flip in the respective (ith) main graph to simulate
    # erroneous estimation; the other XY-plane (the value of (i, j, 1)
    # for a given i, j) specifies the number of random graphs to be
    # made.

    # prob need to rework :/
    # need to keep track of the main_graph for making the erroneous_graph
    pass


def main_graph(?):
    # could just return one, so input is just num vars
    # or could return an array of (random) graphs of the same num_vars
    pass


# rather think of how the results will be displayed: probably a table?
# so the columns would be num_vars, num_reversals (or both num_added,
# num_removed), (avg)_num_latents; we do random reversals and then
# report averages per main graph? or rather average over main graphs
# (of the same size)? and then perhaps give the max number of latents
# (in an mlm) for the given number vars?

# remember, theres 2^(n-1) graphs for n variables
#  
