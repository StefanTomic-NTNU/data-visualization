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
    """ Calculates euclidean distances """
    # matrix2 = np.zeros(shape=matrix.shape)
    # top = np.argpartition(matrix, k, axis=1)[k]
    # for i in range(0, len(top)):
    #     print(top[i])
    #     row = matrix[i]
    #     row[row > top[i]] = 0
    #     print(row)
    #     matrix2[i] = row

    return matrix * (matrix >= np.sort(matrix, axis=1)[:, [k]]).astype(int)
