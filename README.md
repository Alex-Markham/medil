## MeDIL
This package is for causal inference using the Measurement Dependence Inducing Latent (MeDIL) Causal Model framework[^fn1]. 
This package is a work in progress, and there are many features I plan but have yet to implement, but doing so will take some time, especially considering I am primarily a theoritician and not a software engineer.
More information can be found in the [documentation](https://medil.causal.dev) or on the [project web page](https://causal.dev/projects/medil)

---
- [1. Installation](#1-installation)
- [2. Basic usage](#2-basic-usage)
- [3. Support](#3-support)
- [4. Contributing](#4-contributing)
- [5. License](#5-license)
- [6. Changelog](#6-changelog)
- [7. References](#7-references)
---


### 1. Installation

### 2. Basic usage
Example of how I used it on Big 5 data:

```python
import numpy as np
from independence_test import independence_test


# load BIG5 data
b5_data = np.loadtxt('../../BIG5/data.csv', delimiter='\t',
                     skiprows=1, usecols=np.arange(7, 57)).T
with open('../BIG5/data.csv') as file:
    b5_fields = np.asarray(file.readline().split('\t')[7:57])
b5_fields[-1] = b5_fields[-1][:-1]

# run tests
dependencies, p_values, null_hyp = independence_test(b5_data, 10, alpha=.05)
np.savez('perm_test_10.npz', null_hyp=null_hyp, dependencies=dependencies)
```

That was just for the independence permutation test using dist_corr.

-------------------------------------------------------------------------------

Example creating an undirected dependency graph:

```python
>>> from medil import graph


>>> num_vertices = 5
>>> example_UDG = medil.graph.UndirectedDependenceGraph(num_vertices)
>>> example_UDG.adj_matrix
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]])
>>> edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 1]])
>>> example_UDG.add_edges(edges)
>>> example_UDG.adj_matrix
array([[0, 1, 0, 0, 0],
       [1, 0, 1, 0, 1],
       [0, 1, 0, 1, 0],
       [0, 0, 1, 0, 1],
       [0, 1, 0, 1, 0]])
>>> example_UDG.num_edges
5
```

### 3. Support
If you have any questions, suggestions, feedback, or bugs to report, please [contact me](https://causal.dev/#contact) or [open an issue](https://gitlab.com/alex-markham/medil/issues).
Additionally, if you would like to use this package or any of its code in a project, feel free (but not obliged) to contact me.

### 4. Contributing
If you would like to contribute, you can [contact me](https://causal.dev/#contact) and fork this repo on GitLab (or see [this discussion](https://gist.github.com/DavideMontersino/810ebaa170a2aa2d2cad) for information on forking to GitHub).

### 5. License
Refer to [LICENSE](https://gitlab.com/alex-markham/medil/blob/master/LICENSE), which is the [GNU General Public License v3 (GPLv3)](https://choosealicense.com/licenses/gpl-3.0/).

### 6. Changelog
Refer to [CHANGELOG](https://gitlab.com/alex-markham/medil/blob/master/CHANGELOG.md) to see planned features and history of the already implemented features.

### 7. References
[^fn1]: Alex Markham and Moritz Grosse-Wentrup (2019). *Measurement Dependence Inducing Latent Causal Models*. arXiv:1910.08778 [stat.ML]. ([link](https://arxiv.org/abs/1910.08778))
