import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree
import copy

y0_np = np.array((1., 1., 1.)).flatten()
y0_jnp = jnp.array((1., 1., 1.)).flatten()

# Define the Lorenz-63 System
def Lorenz_np(y, params):
    dx1 = params["sigma"] * (y[1] - y[0])
    dx2 = y[0] * (params["rho"] - y[2]) - y[1]
    dx3 = y[0] * y[1] - params["beta"] * y[2]
    return np.array((dx1, dx2, dx3), dtype = object).flatten()

# Euler's Method
def Euler_np(params):
    y = y0_np
    ys = []
    for _  in range(params["length"]): 
        y = y + params["dt"] * Lorenz_np(y, params)
        ys.append(y)
    return np.array(ys).transpose()

# Define the Lorenz-63 System
@jax.jit
def Lorenz_jnp(y, params):
    dx1 = params["sigma"] * (y[1] - y[0])
    dx2 = y[0] * (params["rho"] - y[2]) - y[1]
    dx3 = y[0] * y[1] - params["beta"] * y[2]
    return jnp.array((dx1, dx2, dx3)).flatten()

# Euler's Method
def Euler_jnp(params):
    y = y0_jnp
    ys = []
    for _  in range(params["length"]): 
        y = y + params["dt"] * Lorenz_jnp(y, params)
        ys.append(y)
    return jnp.array(ys).transpose()

params_fixed = {
    "sigma": 10,
    "beta": 1.5, 
    "rho": 40,
    "length": int(1e05),
    "dt": .005
}

def change_param(new_value, param_to_change, params_old = params_fixed):
    params_new = copy.deepcopy(params_old)
    params_new[param_to_change] = new_value
    return params_new

#Contruct Markov matrix
def Markov_np(ys, sample, nt=1):

    # # neighbor searching
    # Xtrain = ys.T
    # distances = np.linalg.norm(sample - Xtrain[:, None, :], axis=2)
    # point_idxs = np.argmin(distances, axis=1)

    tree = cKDTree(sample)
    point_dst, point_idxs = tree.query(ys.T) 

    lp = len(point_idxs)

    transitions = point_idxs[nt:lp] - point_idxs[0:(lp - nt)]
    cellEdge = np.flatnonzero(transitions) + 1

    # # Build markov matrix
    # M_mat = np.zeros((len(sample), len(sample)))
    # ce = cellEdge[:-nt]
    # indices = (point_idxs[ce], point_idxs[ce + nt])
    # M_mat[indices] += 1

    M_mat = np.zeros((len(sample), len(sample)))
    for ind in range(0, len(cellEdge) - nt):
        ce = cellEdge[ind]
        M_mat[point_idxs[ce],point_idxs[ce+nt]] += 1

    # Normalize rows
    row_sums = np.sum(M_mat, axis=1)
    nonzero_rows = np.nonzero(row_sums)
    M_mat[:, nonzero_rows] /= row_sums[nonzero_rows]
        
    # row_sums = np.sum(M_mat2, axis=1)
    # for i in range(subset_size):
    #     if row_sums[i] != 0:
    #         M_mat2[:,i] = M_mat2[:,i]/row_sums[i]

    return M_mat

#Contruct Markov matrix
def Markov_jnp(ys, sample, nt=1):

    # neighbor searching
    Xtrain = ys.T
    distances = jnp.linalg.norm(sample - Xtrain[:, None, :], axis=2)
    point_idxs = jnp.argmin(distances, axis=1)

    # tree = cKDTree(sample)
    # point_dst, point_idxs = tree.query(ys.T) 

    lp = len(point_idxs)

    transitions = point_idxs[nt:lp] - point_idxs[0:(lp - nt)]
    cellEdge = jnp.flatnonzero(transitions) + 1

    # Build markov matrix
    # M_mat = jnp.zeros((len(sample), len(sample)))
    # ce = cellEdge[:-nt]
    # indices = (point_idxs[ce], point_idxs[ce + nt])
    # M_mat = M_mat.at[indices].add(1.)

    M_mat = jnp.zeros((len(sample), len(sample)))
    for ind in range(0, len(cellEdge) - nt):
        ce = cellEdge[ind]
        M_mat = M_mat.at[point_idxs[ce],point_idxs[ce+nt]].add(1)
        
    # Normalize rows
    row_sums = jnp.sum(M_mat, axis=1)
    nonzero_rows = jnp.nonzero(row_sums)
    M_mat = M_mat.at[:, nonzero_rows].divide(row_sums[nonzero_rows])

    return M_mat