Tutorial
========

For a basic overview, see the `software demonstration <https://causal.dev/files/medil_demo.pdf>`_ :cite:`Markham_2020_PGM` presented at the 10th International Conference on Probabilistic Graphical Models (PGM) and the associated script, which you can see below or download `here <https://gitlab.com/alex-markham/medil/-/raw/develop/scripts/pgm_demo.py?inline=false>`_:

.. literalinclude:: ../scripts/pgm_demo.py
   :linenos:
   :language: python

..
   The most basic use case of ``medil`` starts with a given data set and has two main steps: (1) using independence testing to estimate the undirected dependency graph (UDG) and (2) using this UDG to find the structural MeDIL Causal Model (MCM).
   Continue on to the `Advanced Tutorial`_ for further topics, like `data simulation`_, `nonlinear independence testing`_ and `learning functional MCMs`_.

   ::

      >>> import numpy as np
      >>> from medil.independence_testing import hypothesis_test
      >>> from medil.ecc_algorithms import find_clique_min_cover as find_cm


   Use distance correlation as a non-linear measure of dependence for permutation based hypothesis testing (could take a few minutes, depending on your machine)::

     >>> p_vals, null_corr, dep_graph = hypothesis_test(m_samps, 100, )
     >>> dep_graph = dependencies(null_corr, .1, p_vals, .05)
     >>> dep_graph
     array([[ True,  True,  True, False, False, False],
	    [ True,  True,  True,  True,  True, False],
	    [ True,  True,  True, False,  True,  True],
	    [False,  True, False,  True,  True, False],
	    [False,  True,  True,  True,  True,  True],
	    [False, False,  True, False,  True,  True]])


   find the edges of a MeDIL causal model with the fewest number of latent variables::

     >>> min_mcm = find_cm(dep_graph)
     >>> min_mcm
     array([[0, 0, 1, 0, 1, 1],
	    [0, 1, 0, 1, 1, 0],
	    [1, 1, 1, 0, 0, 0]])


   The result is a directed biadjacency matrix, where rows are latents variables, columns are measurement variables, and each 1 indicates a directed edge from the latent to measurement variable.

   Advanced Tutorial
   -----------------


   Data Simulation
   ~~~~~~~~~~~~~~~
   Linear case:
   Simulate 1000 samples of data for 6 measurement variables with dependencies induced by 3 latent variables::

     >>> num_samps = 1000
     >>> m_noise = np.random.standard_normal((6, num_samps))
     >>> l_noise = np.random.standard_normal((3, num_samps))
     >>> 
     >>> m_samps = np.empty((6, num_samps))
     >>> m_samps[0] = l_noise[0] + .2 * m_noise[0]
     >>> m_samps[1] = 2 * l_noise[0] - 3 *l_noise[1]  + m_noise[1]
     >>> m_samps[2] = 5 * l_noise[2] +  m_noise[2] - 10 * l_noise[0]
     >>> m_samps[3] = 4 * l_noise[1] + .5 * m_noise[3]
     >>> m_samps[4] = l_noise[2] * 3 + l_noise[1] * 3 + m_noise[4]
     >>> m_samps[5] = -7 * l_noise[2] + m_noise[5] * 2
     >>> m_samps
     array([[ -1.09618589,   0.36562761,   0.1821539 , ...,  -1.15422013,   -0.47157367,   0.86087563],
	    [  3.58650565,   1.01922667,   3.038765  , ...,  -2.30804292,    5.83232892,   2.08316629],
	    [ 14.13299282,   3.21167764, -17.29291657, ...,  14.83832827,   -6.53695807,  -9.48083731],
	    [ -6.62993973,   1.34280329,  -2.1832981 , ...,  -0.35295787,   -5.6925398 ,   0.531198  ],
	    [ -0.96126815,   2.93188441, -11.50689577, ...,   0.98983901,   -8.5647367 ,  -0.14181254],
	    [-12.74328352,  -4.95887306,  20.20499673, ...,  -0.49624485,    6.5901562 ,   0.08445142]])


   Nonlinear case using GANs:


   Nonlinear Independence Testing
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



   Learning Functional MCMs
   ~~~~~~~~~~~~~~~~~~~~~~~~
