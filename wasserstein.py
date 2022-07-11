import numpy as np
def distance(u_values, v_values, u_weights=None, v_weights=None):
    u_values, u_weights = np.asarray(u_values, dtype=float), np.asarray(u_weights, dtype=float)
    v_values, v_weights = np.asarray(v_values, dtype=float), np.asarray(v_weights, dtype=float)
    # u_values==[0. 1. 2. 3.], v_sorter==[0. 1. 2. 3.]

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)
    # u_sorter==[0 1 2 3], v_sorter==[0 1 2 3]

    all_values = np.concatenate((u_values, v_values))    
    all_values.sort(kind='mergesort')
    # all_values==[0. 0. 1. 1. 2. 2. 3. 3.]

    deltas = np.diff(all_values)
    # deltas==[0. 1. 0. 1. 0. 1. 0.]

    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')
    # u_cdf_indices==[1 1 2 2 3 3 4]
    # v_cdf_indices==[1 1 2 2 3 3 4]

    u_sorted_cumweights = np.concatenate(([0], np.cumsum(u_weights[u_sorter])))
    u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]
    # u_sorted_cumweights==[0. 1. 1. 1. 1.] 
    # u_cdf==[1. 1. 1. 1. 1. 1. 1.]
    
    v_sorted_cumweights = np.concatenate(([0], np.cumsum(v_weights[v_sorter])))
    v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]
    # v_sorted_cumweights==[0. 0. 0. 0. 1.] 
    # v_cdf==[0. 0. 0. 0. 0. 0. 1.]
    return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))

from scipy import stats
u = [1,0,0,0]
v = [0,0,0,1]
dists = range(len(u))
print(stats.wasserstein_distance([0,1,2,3],[0,1,2,3], u, v))
print(distance([0,1,2,3],[0,1,2,3], u, v))