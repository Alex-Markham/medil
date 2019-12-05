import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering


## psuedo code:
# 1. fix sample order for n samples of k features
# 2. compute a n-by-k_choose_2 'contribution' matrix where c_ij is the contribution of sample i to the pairwaise correlation which would be the sum of column j
# 3. do PCA on contribution matrix


## sample data, 100 samps

# latents
l_0 = np.random.randn(50, 1)
l_1 = np.random.randn(30, 1)
l_2 = np.random.randn(20, 1)

# convenient for analysis to have actual mask instead of just using np.shuffle() on data
pre_mask = np.zeros(100, int)
pre_mask[50:80] = 1
pre_mask[80:] = 2
mask = np.random.permutation(pre_mask)


# innovations
x = np.random.randn(100, 15)

# include latents
x[mask==0, :7] += 2 * l_0
x[mask==1, 5:12] += 2 * l_1 
x[mask==2, 10:] += 2 * l_2

# following covs should confirm intended structure
pooled = np.corrcoef(x, rowvar=False)
just_l_0 = np.corrcoef(x[mask==0], rowvar=False)
just_l_1 = np.corrcoef(x[mask==1], rowvar=False)
just_l_2 = np.corrcoef(x[mask==2], rowvar=False)


## on to clustering

# 2. find sample contribution to covs
contribution = np.zeros((100, 105))  # second num is k choose 2

means = x.mean(0)
stds = x.std(0)

con_idxs = np.triu_indices(15, 1)  # first num is k
con_xvar_idxs = zip(con_idxs[0], con_idxs[1])

for idx, pair in enumerate(con_xvar_idxs):
    contribution[:, idx] = np.divide((np.multiply((x[:, pair[0]] - means[pair[0]]), (x[:, pair[1]] - means[pair[1]])) / 100), (stds[pair[0]] * stds[pair[1]]))

    
# 3. PCA on cons
pca = PCA(n_components=5)
principal_contribution = pca.fit_transform(contribution)
sorted_samp_idcs = np.argsort(principal_contribution, axis=0)[::-1]
print(mask[sorted_samp_idcs[:,0]])
print(mask[sorted_samp_idcs[:,1]])
print(mask[sorted_samp_idcs[:,2]])
print(mask[sorted_samp_idcs[:,3]])
print(mask[sorted_samp_idcs[:,4]])
print(pca.explained_variance_ratio_)

## sliding plot


## cov heatmaps

# fig, sub = plt.subplots(2, 4)
# sns.heatmap(pooled, square=True, ax=sub[0, 0])
# sub[0,0].title.set_text("all samples")
# sns.heatmap(just_l_0, square=True, ax=sub[0, 1])
# sub[0, 1].title.set_text("samples from L_0")
# sns.heatmap(just_l_1, square=True, ax=sub[0, 2])
# sub[0, 2].title.set_text("samples from L_1")
# sns.heatmap(just_l_2, square=True, ax=sub[0, 3])
# sub[0, 3].title.set_text("samples from L_2")

# sns.heatmap(cons_cov, ax=sub[1, 0])
# sub[1, 0].title.set_text("contribution")
# sns.heatmap(eig_vecs, ax=sub[1, 1])
# sub[1, 1].title.set_text("eigenvectors")
# sns.heatmap(eig_vecs[:, sorted_idx[:5]], ax=sub[1, 2])
# sub[1, 2].title.set_text("top 5 eigenvectors, sorted")
# sns.heatmap(projected_cons, ax=sub[1, 3])
# sub[1, 3].title.set_text("contributions projected to top 5 eignvectors, sorted")
# plt.show()


## 2 component clusters
plt.scatter(principal_contribution[:, 0], principal_contribution[:,1], c=mask)
plt.show()

## try spectral clustering using cons_cov (or a transformation of it) as affinity matrix
affinity_mat = np.abs(np.cov(contribution))
clustering = SpectralClustering(3, affinity='precomputed')
clusters = clustering.fit_predict(affinity_mat)

print(clusters[sorted_samp_idcs[:,0]])
print(clusters[sorted_samp_idcs[:,1]])
print(clusters[sorted_samp_idcs[:,2]])
print(clusters[sorted_samp_idcs[:,3]])
print(clusters[sorted_samp_idcs[:,4]])

# group labels may (of course) be different, but there's always a clear equivalence between the PCA groups and the spectral clustering groups 
