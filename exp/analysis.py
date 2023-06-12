from gloabl_settings import DATA_PATH
from exp.examples import paths_list
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
sns.set()


def analysis(biadj_mat, biadj_mat_recon):
    """Perform analysis of the distances between true and reconstructed structures
    Parameters
    ----------
    biadj_mat: input directed graph
    biadj_mat_recon: learned directed graph in the form of adjacency matrix

    Returns
    -------
    sfd: squared Frobenius distance (bipartite graph)
    ushd: structural hamming distance (undirected graph)
    """

    # ushd = shd_func(recover_ug(biadj_mat), recover_ug(biadj_mat_recon))
    ug = recover_ug(biadj_mat)
    ug_recon = recover_ug(biadj_mat_recon)

    ushd = np.triu(np.logical_xor(ug, ug_recon), 1).sum()

    biadj_mat = biadj_mat.astype(int)
    biadj_mat_recon = biadj_mat_recon.astype(int)

    wtd_ug = biadj_mat.T @ biadj_mat
    wtd_ug_recon = biadj_mat_recon.T @ biadj_mat_recon

    sfd = ((wtd_ug - wtd_ug_recon) ** 2).sum()

    return sfd, ushd


def recover_ug(biadj_mat):
    """Recover the undirected graph from the directed graph
    Parameters
    ----------
    biadj_mat: learned directed graph

    Returns
    -------
    ug: the recovered undirected graph
    """

    # get the undirected graph from the directed graph
    ug = biadj_mat.T @ biadj_mat
    np.fill_diagonal(ug, False)

    return ug


def build_table(n, p):
    """Build table for SHD, ELBO, and losses
    Parameters
    ----------
    n: number of observed variables
    p: edge probability

    Returns
    -------
    table: table for summarizing the results
    """

    exp_path = os.path.join(DATA_PATH, "experiments")
    columns = [
        "loss_true_train",
        "loss_true_valid",
        "error_true_train",
        "error_true_valid",
        "loss_recon_train",
        "loss_recon_valid",
        "error_recon_train",
        "error_recon_valid",
        "loss_vanilla_train",
        "loss_vanilla_valid",
        "error_vanilla_train",
        "error_vanilla_valid",
        "shd_recon"
    ] + ["train_flag", "valid_flag"] + ["dof", "run"]

    table = pd.DataFrame(columns=columns)

    for idx in range(10):
        sub_table = pd.DataFrame(pd.NA, index=paths_list, columns=columns)

        for path in paths_list:
            graph_path = os.path.join(exp_path, f"experiment_{idx}", path)
            num = path.split("_")[1]
            if num in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                result_path = os.path.join(graph_path, f"n={n}_p={p}")
            else:
                result_path = os.path.join(graph_path)

            # hrstc graph
            loss_recon = pd.read_pickle(os.path.join(result_path, "loss_recon.pkl"))
            error_recon = pd.read_pickle(os.path.join(result_path, "error_recon.pkl"))
            sub_table.loc[path, "loss_recon_train"] = loss_recon[0][-1]
            sub_table.loc[path, "loss_recon_valid"] = loss_recon[1][-1]
            sub_table.loc[path, "error_recon_train"] = error_recon[0][-1]
            sub_table.loc[path, "error_recon_valid"] = error_recon[1][-1]

            # vanilla graph
            loss_vanilla = pd.read_pickle(os.path.join(result_path, "loss_vanilla.pkl"))
            error_vanilla = pd.read_pickle(
                os.path.join(result_path, "error_vanilla.pkl")
            )
            sub_table.loc[path, "loss_vanilla_train"] = loss_vanilla[0][-1]
            sub_table.loc[path, "loss_vanilla_valid"] = loss_vanilla[1][-1]
            sub_table.loc[path, "error_vanilla_train"] = error_vanilla[0][-1]
            sub_table.loc[path, "error_vanilla_valid"] = error_vanilla[1][-1]

            if "Graph" in path:
                # true graph
                loss_true = pd.read_pickle(os.path.join(result_path, "loss_true.pkl"))
                error_true = pd.read_pickle(os.path.join(result_path, "error_true.pkl"))
                sub_table.loc[path, "loss_true_train"] = loss_true[0][-1]
                sub_table.loc[path, "loss_true_valid"] = loss_true[1][-1]
                sub_table.loc[path, "error_true_train"] = error_true[0][-1]
                sub_table.loc[path, "error_true_valid"] = error_true[1][-1]

                # SHD for reconstruction
                biadj_mat = np.load(os.path.join(result_path, "biadj_mat.npy"))
                biadj_mat_recon = np.load(os.path.join(result_path, "biadj_mat_recon.npy"))
                shd, ushd = analysis(biadj_mat, biadj_mat_recon)
                sub_table.loc[path, "shd_recon"] = shd

            # performance information
            train_flag = table["loss_recon_train"] < table["loss_vanilla_train"]
            valid_flag = table["loss_recon_valid"] < table["loss_vanilla_valid"]
            boolean_dictionary = {True: "recon", False: "vanilla"}
            table["train_flag"] = train_flag.map(boolean_dictionary)
            table["valid_flag"] = valid_flag.map(boolean_dictionary)

            # other information
            info = pd.read_pickle(os.path.join(result_path, "info.pkl"))
            biadj_mat = np.load(os.path.join(result_path, "biadj_mat.npy"))
            _, num_obs = biadj_mat.shape
            if info["dof"] is None:
                dof = num_obs**2 // 4
            else:
                dof = info["dof"]
            sub_table["dof"] = dof
            sub_table["run"] = idx

        table = pd.concat([table, sub_table])

    return table


def plot_diff(graph_num, obs, density):
    """ Plot the differences
    Parameters
    ----------
    graph_num: graph number
    obs: number of observations
    density: density of the graphs
    """

    # create dictionary of losses
    exp_path = os.path.join(DATA_PATH, "experiments")
    columns = [f"Train-$\Delta$-Baseline"] + [f"Valid-$\Delta$-Baseline"] + \
              [f"Train-$\Delta$-True"] + [f"Valid-$\Delta$-True"]

    df = pd.DataFrame(index=range(10), columns=columns)
    for exp_num in range(10):
        graph_path = os.path.join(exp_path, f"experiment_{exp_num}", f"Graph_{graph_num}")
        obs_path = os.path.join(graph_path, f"n={obs}_p={density}")
        vnl = pd.read_pickle(os.path.join(obs_path, "loss_vanilla.pkl"))
        rec = pd.read_pickle(os.path.join(obs_path, "loss_recon.pkl"))
        true = pd.read_pickle(os.path.join(obs_path, "loss_true.pkl"))
        df.loc[exp_num, f"Train-$\Delta$-Baseline"] = vnl[0] - rec[0]
        df.loc[exp_num, f"Valid-$\Delta$-Baseline"] = vnl[1] - rec[1]
        df.loc[exp_num, f"Train-$\Delta$-True"] = true[0] - rec[0]
        df.loc[exp_num, f"Valid-$\Delta$-True"] = true[1] - rec[1]
    dic = {column: df[column].mean() for column in df.columns}

    # plot and save figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for idx, (key, value) in enumerate(dic.items()):
        linestyle = "-" if "Train" in key else "--"
        ax.plot(value, color=sns.color_palette()[idx % 8], linestyle=linestyle, label=key)
    ax.legend(loc="upper right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Difference")
    ax.set_title(f"Differences in losses between Baseline/True and NCFA loss for p={density}")
    fig.savefig(os.path.join(exp_path, "diff", f"Graph_{graph_num}.pdf"), bbox_inches="tight")


def plot_learning(graph_num, obs, density):
    """ Plot the differences
    Parameters
    ----------
    graph_num: graph number
    obs: number of observations
    density: density of the graphs
    """

    # create dictionary of losses
    exp_path = os.path.join(DATA_PATH, "experiments")
    columns = [f"Train-NCFA"] + [f"Valid-NCFA"] + \
              [f"Train-True"] + [f"Valid-True"] + \
              [f"Train-Baseline"] + [f"Valid-Baseline"]

    df = pd.DataFrame(index=range(10), columns=columns)
    for exp_num in range(10):
        graph_path = os.path.join(exp_path, f"experiment_{exp_num}", f"Graph_{graph_num}")
        obs_path = os.path.join(graph_path, f"n={obs}_p={density}")
        vnl = pd.read_pickle(os.path.join(obs_path, "loss_vanilla.pkl"))
        rec = pd.read_pickle(os.path.join(obs_path, "loss_recon.pkl"))
        true = pd.read_pickle(os.path.join(obs_path, "loss_true.pkl"))
        df.loc[exp_num, f"Train-NCFA"] = rec[0]
        df.loc[exp_num, f"Valid-NCFA"] = rec[1]
        df.loc[exp_num, f"Train-Baseline"] = vnl[0]
        df.loc[exp_num, f"Valid-Baseline"] = vnl[1]
        df.loc[exp_num, f"Train-True"] = true[0]
        df.loc[exp_num, f"Valid-True"] = true[1]
    dic = {column: df[column].mean() for column in df.columns}

    # plot and save figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for idx, (key, value) in enumerate(dic.items()):
        linestyle = "-" if "Train" in key else "--"
        ax.plot(value, color=sns.color_palette()[idx % 8], linestyle=linestyle, label=key)
    ax.legend(loc="upper right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Losses")
    ax.set_title(f"Losses of NCFA, Baseline, and True for p={density}")
    fig.savefig(os.path.join(exp_path, "diff", f"Graph_{graph_num}.pdf"), bbox_inches="tight")
