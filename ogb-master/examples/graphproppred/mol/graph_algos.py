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


def floyd_warshall(adjacency_matrix, max_dist=9999):
    (mat, n) = check_and_convert_adjacency_matrix(adjacency_matrix)

    for k in range(n):
        mat = minimum(mat, mat[newaxis, k, :] + mat[:, k, newaxis])

    mat = torch.Tensor(mat)

    unreachable_index = mat == inf
    mat[mat > max_dist] = far_away_marker(max_dist)
    mat[unreachable_index] = unreachable_marker(max_dist)

    # slow
    return mat.numpy()


def unreachable_marker(max_dist):
    return max_dist + 2


def far_away_marker(max_dist):
    return max_dist + 1
