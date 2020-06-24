import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import SpectralClustering


## psuedo code:
# 1. fix sample order for n samples of m features
# 2. compute a n-by-m_choose_2 'contribution' matrix where c_ij is the contribution of sample i to the pairwaise correlation which would be the sum of column j
# 3. use n-by-n covariance of contribution matrix as affinity matrix for spectral clustering (equivalent to doing PCA on contribution matrix)


def make_linear_contribution_matrix(data):
    num_samples, num_vars = data.shape
    num_vars_choose_2 = num_vars * (num_vars - 1) // 2

    # init contr_mat
    contr_mat = np.zeros((num_samples, num_vars_choose_2))

    # fill in contributions
    means = data.mean(0)
    stds = data.std(0)

    con_idxs = np.triu_indices(num_vars, 1)
    con_var_idxs = zip(con_idxs[0], con_idxs[1])

    for idx, pair in enumerate(con_var_idxs):
        contr_mat[:, idx] = np.divide((np.multiply((data[:, pair[0]] - means[pair[0]]), (data[:, pair[1]] - means[pair[1]])) / num_samples), (stds[pair[0]] * stds[pair[1]]))
    return contr_mat

    
def pca_clusters(contribution_matrix, n_components):
    pca = PCA(n_components=n_components)
    principal_contribution = pca.fit_transform(contribution_matrix)
    sorted_samp_idcs = np.argsort(principal_contribution, axis=0)[::-1]
    return principal_contribution, sorted_samp_idcs


def ica_clusters(contribution_matrix, n_components):
    ica = FastICA(n_components=n_components)
    indep_compon_contributions = ica.fit_transform(contribution_matrix)
    sorted_samp_idcs = np.argsort(indep_compon_contributions, axis=0)[::-1]
    return indep_compon_contributions, sorted_samp_idcs


def spectral_clusters(num_clusters=None):
    affinity_mat = np.abs(np.cov(contribution))

    if num_clusters is None:
        clustering = SpectralClustering(affinity='precomputed')
    else:
        clustering = SpectralClustering(num_clusters, affinity='precomputed')

    return clustering.fit_predict(affinity_mat)
