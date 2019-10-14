import matplotlib.pyplot as plt
import numpy as np


results = np.load('../../BIG5/big_5_p_vals_4resamps.npz')
p_vals = results['p_vals']
null_cov = results['null_cov']
perm_covs = results['perm_covs']

plt.imshow(p_vals, cmap='hot', interpolation='nearest')
