import torch
from numpy import array, asarray, inf, zeros, minimum, diagonal, newaxis
from numpy.random import randint
import time


def check_and_convert_adjacency_matrix(adjacency_matrix):
    if isinstance(adjacency_matrix, torch.Tensor):
        adjacency_matrix = adjacency_matrix.numpy()

    mat = asarray(adjacency_matrix)

    (nrows, ncols) = mat.shape
    assert nrows == ncols
    n = nrows

    assert (diagonal(mat) == 0.0).all()

    return (mat, n)


def floyd_warshall(adjacency_matrix, max_dist=None):
    (mat, n) = check_and_convert_adjacency_matrix(adjacency_matrix)

    for k in range(n):
        mat = minimum(mat, mat[newaxis, k, :] + mat[:, k, newaxis])

    assert mat.max() < inf, 'expecting connected graph'

    mat = torch.Tensor(mat)
    if max_dist:
        mat[mat > max_dist] = max_dist + 1

    return mat
