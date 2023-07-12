import numpy as np
import jax
import jax.numpy as jnp
import copy

# Define the Lorenz-63 System
@jax.jit
def Lorenz(y, params):
    dx1 = params["sigma"] * (y[1] - y[0])
    dx2 = y[0] * (params["rho"] - y[2]) - y[1]
    dx3 = y[0] * y[1] - params["beta"] * y[2]
    return jnp.array((dx1, dx2, dx3)).flatten()

# Euler's Method
def Euler(params):
    y = params["y0"]
    ys = []
    for _  in range(params["length"]): 
        y = y + params["dt"] * Lorenz(y, params)
        ys.append(y)
    return jnp.array(ys).transpose()

def change_param(params_old, new, param_to_change):
    params_new = copy.deepcopy(params_old)
    params_new[param_to_change] = new
    return params_new

#Contruct Markov matrix
def Markov(ys, sample, nt=1):

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
    M_mat = jnp.zeros((len(sample), len(sample)))
    ce = cellEdge[:-nt]
    indices = (point_idxs[ce], point_idxs[ce + nt])
    M_mat = M_mat.at[indices].add(1.)

    # Normalize rows
    row_sums = jnp.sum(M_mat, axis=1)
    nonzero_rows = jnp.nonzero(row_sums)
    M_mat = M_mat.at[:, nonzero_rows].divide(row_sums[nonzero_rows])

    return M_mat