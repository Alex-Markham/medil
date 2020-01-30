import numpy as np

# test independence estimation
from medil.independence_testing import hypothesis_test
from medil.independence_testing import dependencies


def test_indep_testing():
    num_samps = 1000
    m_noise = np.random.standard_normal((6, num_samps))
    # l_0 = np.random.triangular(5, 9, 10, num_samps)
    # l_1 = np.random.laplace(-7, 100, num_samps)
    # l_2 = np.random.random(num_samps)
    
    # m_samps = np.empty((6, num_samps))
    # m_samps[0] = l_0 * m_noise[0]
    # m_samps[1] = np.cos(l_0) - l_1 * np.abs(m_noise[1])
    # m_samps[2] = l_2 * np.log(m_noise[2] ** 2) - l_0
    # m_samps[3] = l_1 + m_noise[3]
    # m_samps[4] = np.sqrt(np.abs(l_2 * l_1)) * m_noise[4]
    # m_samps[5] = l_2  + m_noise[5] * 10
    
    l_noise = np.random.standard_normal((3, num_samps))
    
    m_samps = np.empty((6, num_samps))
    m_samps[0] = l_noise[0] + .2 * m_noise[0]
    m_samps[1] = 2 * l_noise[0] - 3 *l_noise[1]  + m_noise[1]
    m_samps[2] = 5 * l_noise[2] +  m_noise[2] - 10 * l_noise[0]
    m_samps[3] = 4 * l_noise[1] + .5 * m_noise[3]
    m_samps[4] = l_noise[2] * 3 + l_noise[1] * 3 + m_noise[4]
    m_samps[5] = -7 * l_noise[2] + m_noise[5] * 2
    
    p_vals, null_corr = hypothesis_test(m_samps, 100)
    dep_graph = dependencies(null_corr, .1, p_vals, .05)
    
    # assert something in relation to pearson corr?
    # definitely test if individual pairs in the joint matrix have same values as if done only on pair of variables
    # maybe also test halfing num tests because of assymetry 
