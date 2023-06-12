from exp.examples import fixed_biadj_mat_list, conversion_dict
from exp.examples import tcga_key, mnist_key, tumors_key
from sklearn.preprocessing import StandardScaler
from gloabl_settings import DATA_PATH
from exp.pipeline import pipeline_graph
from exp.pipeline import pipeline_real
from datetime import datetime
import pandas as pd
import os


def run_fixed(num_samps_graph, heuristic, method, alpha, dof, dof_method, exp_path, seed):
    """ Run MeDIL on the fixed graphs
    Parameters
    ----------
    num_samps_graph: number of samples for graph
    heuristic: whether to use heuristic or not
    method: method for udg estimation
    alpha: alpha value
    dof: desired size of latent space of VAE
    dof_method: how to distribute excess degrees of freedom to latent causal factors
    exp_path: path for the experiment
    seed: random seed for the experiments
    """

    for idx, biadj_mat in enumerate(fixed_biadj_mat_list):
        graph_idx = conversion_dict[idx]
        graph_path = os.path.join(exp_path, f"Graph_{graph_idx}")

        if not os.path.isdir(graph_path):
            os.mkdir(graph_path)

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on graph {graph_idx} with "
              f"num_samps={num_samps_graph}")
        pipeline_graph(
            biadj_mat, num_samps_graph, heuristic, method, alpha, dof, dof_method, graph_path, seed=seed
        )


def run_random1(num_samps_graph, heuristic, method, alpha, dof, dof_method, exp_path, seed):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    num_samps_graph: number of samples for graph
    heuristic: whether to use heuristic or not
    method: method for udg estimation
    alpha: alpha value
    dof: desired size of latent space of VAE
    dof_method: how to distribute excess degrees of freedom to latent causal factors
    exp_path: path for the experiment
    seed: random seed for the experiments
    """

    for key, biadj_mat in rand_biadj_mat_list1.items():
        idx, n, p = key.split("_")
        graph_path = os.path.join(exp_path, f"Graph_{idx}")
        if not os.path.isdir(graph_path):
            os.mkdir(graph_path)

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on graph {idx} with "
              f"num_samps={num_samps_graph}, n={n}, p={p}")
        folder_name = f"n={n}_p={p}"
        folder_path = os.path.join(graph_path, folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
            pipeline_graph(
                biadj_mat, num_samps_graph, heuristic, method, alpha, dof, dof_method, folder_path, seed=seed
            )


def run_random2(num_samps_graph, heuristic, method, alpha, dof, dof_method, exp_path, seed):
    """ Run MeDIL on the random graphs
    Parameters
    ----------
    num_samps_graph: number of samples for graph
    heuristic: whether to use heuristic or not
    method: method for udg estimation
    alpha: alpha value
    dof: desired size of latent space of VAE
    dof_method: how to distribute excess degrees of freedom to latent causal factors
    exp_path: path for the experiment
    seed: random seed for the experiments
    """

    for key, biadj_mat in rand_biadj_mat_list2.items():
        idx, num_latent = key.split("_")
        graph_path = os.path.join(exp_path, f"Graph_{idx}")
        if not os.path.isdir(graph_path):
            os.mkdir(graph_path)

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on graph {idx} with "
              f"num_samps={num_samps_graph}, num_latent={num_latent}")
        folder_name = f"num_latent={num_latent}"
        folder_path = os.path.join(graph_path, folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
            pipeline_graph(
                biadj_mat, num_samps_graph, heuristic, method, alpha, dof, dof_method, folder_path, seed=seed
            )


def run_real(dataset_name, num_samps_real, heuristic, method, alpha, dof, dof_method, exp_path, seed):
    """ Run MeDIL on real dataset
    Parameters
    ----------
    dataset_name: name of dataset
    num_samps_real: number of samples for real dataset
    heuristic: whether to use heuristic or not
    method: method for udg estimation
    alpha: alpha value
    dof: desired size of latent space of VAE
    dof_method: how to distribute excess degrees of freedom to latent causal factors
    exp_path: path for the experiment
    seed: random seed for the experiments
    """

    dataset_path = os.path.join(DATA_PATH, "dataset")
    sc = StandardScaler()
    dataset_train = pd.read_csv(os.path.join(dataset_path, f"{dataset_name}_train.csv"))
    dataset_valid = pd.read_csv(os.path.join(dataset_path, f"{dataset_name}_valid.csv"))
    dataset_train = pd.DataFrame(sc.fit_transform(dataset_train), dataset_train.index, dataset_train.columns).values
    dataset_valid = pd.DataFrame(sc.fit_transform(dataset_valid), dataset_valid.index, dataset_valid.columns).values

    if dataset_name == "tcga":
        dataset_key = tcga_key
    elif dataset_name == "mnist":
        dataset_key = mnist_key
    elif dataset_name == "tumors":
        dataset_key = tumors_key
    else:
        raise ValueError("Invalid dataset name")

    dataset_train_sub = dataset_train[:, dataset_key]
    dataset_valid_sub = dataset_valid[:, dataset_key]
    dataset = [dataset_train_sub, dataset_valid_sub]

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on real data with "
          f"num_samps={num_samps_real}")
    pipeline_real(dataset, heuristic, method, alpha, dof, dof_method, exp_path, seed=seed)
