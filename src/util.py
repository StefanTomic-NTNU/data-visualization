""" Contains helper functions """
import numpy as np
from scipy.spatial.distance import pdist, squareform

csv_rel_path = "../csv/"


def load_csv_to_array(filepath):
    """ Loads csv to numpy matrix """
    return np.genfromtxt(csv_rel_path + filepath, delimiter=',')


def calculate_euclidean_distances(matrix):
    """ Calculates euclidean distances """
    return squareform(pdist(matrix, 'euclidean'))


def reduce_matrix(matrix, k):
    """ Keeps only the k closest points """
    indexes = np.argpartition(matrix, k, axis=1)[:, :k]
    top = indexes[k]
    zero_matrix = np.zeros(shape=matrix.shape)
    for i in range(0, matrix.shape[0]):
        zero_matrix[i, indexes[i]] = matrix[i, indexes[i]]

    # print("Values bigger than 0: ")
    # for j in range(0, zero_matrix.shape[0]):
    #     true_row = [zero_matrix[j, :] > 0][0]
    #     print(true_row)
    #     # print(true_row.shape)
    #     print(zero_matrix[j, true_row])

    print("Indexes: ")
    print(indexes)

    print("\nNew Matrix: ")
    print(zero_matrix)

    # values = np.array([matrix[row, i] for row in range(0, indexes.shape[1]) for i in indexes[row]])
    # values = np.array([np.take(matrix, row, axis=1) for row in indexes])
    # print(values)
    # print(values.shape)

    # new_matrix = [matrix[row] for row in indexes]

    # print("New matrix: ")
    # print(new_matrix)
    # np.matmul(matrix, (matrix >= np.sort(matrix, axis=1)[:, [k]]).astype(int))

    return zero_matrix
