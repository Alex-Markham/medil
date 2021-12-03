"""Implementation of the Greedy Unconditional Equivalence Search Algorithm."""
import numpy as np


def find_optimal_DAG(data_or_deps):
    num_rows, num_cols = data_or_deps.shape()
    if num_rows != num_cols:
        pass                    # do test to get deps
    else:
        deps = data_or_deps

    dag = find_initial_dag(deps)
    score = compute_BIC(dag)
    
    while :

    return dag

def find_initial_dag(deps):
    return


def compute_BIC(dag):
    return
