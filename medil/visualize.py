""" Create and display various graphs, plots, and visualizations."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx



# %%
def show_obs_adj_mat(incidence_mat):

    fig, ax = plt.subplots()

    adj_mat = get_adj_from_incidence(incidence_mat)
    plt.imshow(adj_mat)

    dim = adj_mat.shape[0]

    ax.set_xticks(np.arange(0, dim))
    ax.set_xticklabels([str(idx) for idx in np.arange(1, dim + 1)])

    ax.set_yticks(np.arange(0, dim))
    ax.set_yticklabels([str(idx) for idx in np.arange(1, dim + 1)])

    plt.show()


# %%
def show_obs_dcor_mat(dcor_mat, thresh=None, print_val=False):

    fig, ax = plt.subplots()

    if thresh is not None:
        dcor_mat = (dcor_mat > thresh).astype(int)

    plt.imshow(dcor_mat, vmin=0, vmax=1)
    if print_val:
        for (j, i), label in np.ndenumerate(dcor_mat):
            ax.text(i, j, round(label, 2), ha='center', va='center', fontdict={'color': 'w'})

    dim = dcor_mat.shape[0]

    ax.set_xticks(np.arange(0, dim))
    ax.set_xticklabels([f'X{idx}' for idx in np.arange(1, dim + 1)])

    ax.set_yticks(np.arange(0, dim))
    ax.set_yticklabels([f'X{idx}' for idx in np.arange(1, dim + 1)])

    plt.colorbar()
    plt.show()


# %%
def show_dag(biadj_mat):
    """

    :param biadj_mat:
    :return:
    """

    num_latent, num_obs = biadj_mat.shape

    pos_dict = {}

    latent_pos_dict = {idx:(val,1) for idx, val in enumerate(np.linspace(0, 1, num_latent))}
    obs_pos_dict = {idx+num_latent:(val,0) for idx, val in enumerate(np.linspace(0, 1, num_obs))}


    pos_dict.update(latent_pos_dict)
    pos_dict.update(obs_pos_dict)
    # print(pos_dict)

    node_color =[]
    node_color.extend(num_latent*[0])
    node_color.extend(num_obs*[1])

    full_adj_mat = get_dag_from_biadj(biadj_mat)

    G = nx.DiGraph(full_adj_mat)

    nx.draw_networkx(G, pos=pos_dict, with_labels=False, node_size=2350)
    nx.draw_networkx_labels(G, pos=latent_pos_dict, labels={idx: '$L_{{{}}}$'.format(idx) for idx in range(num_latent)}, font_color='w')
    nx.draw_networkx_labels(G, pos=obs_pos_dict, labels={idx+num_latent: '$M_{{{}}}$'.format(idx) for idx in range(num_obs)}, font_color='k')
    nx.draw_networkx_nodes(G, node_size=2500, pos=pos_dict, node_color=node_color)
    # nx.draw_networkx(G, pos=pos_dict, arrows=True, with_labels=False)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.5, 1.5)
    plt.show()

    
# %%
def get_dag_from_biadj(biadj_mat):

    num_latent, num_obs = biadj_mat.shape

    dag_adj_mat = np.zeros((num_latent+num_obs, num_latent+num_obs))

    dag_adj_mat[:num_latent, num_latent:] = biadj_mat

    return dag_adj_mat


# %%
def show_pairwise_plot(sample: np.ndarray, color='C0'):
    """

    :param sample:
    :return:
    """

    if sample.shape[0] > 1000:
        print(f'sample of size {sample.shape[0]} too big, using first 1000.')
        sample = sample[:1000, :]

    sample_df = pd.DataFrame(sample, columns=[f'X{idx}' for idx in np.arange(1, sample.shape[1]+1)])
    sns.pairplot(sample_df, corner=True, diag_kind='kde',
                 plot_kws=dict(color=color), diag_kws=dict(color=color))
    plt.show()
