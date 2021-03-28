""" Contains helper functions """
import numpy as np
from scipy.spatial.distance import pdist, squareform

CSV_RELATIVE_PATH = "../csv/"


def load_csv_to_array(filepath):
    """ Loads csv to numpy matrix """
    return np.genfromtxt(CSV_RELATIVE_PATH + filepath, delimiter=',')


def calculate_euclidean_distances(matrix):
    """ Calculates euclidean distances """
    return squareform(pdist(matrix, 'euclidean'))


def reduce_matrix(matrix, k):
    """ Keeps only the k closest points """
    indexes = np.argpartition(matrix, k, axis=1)[:, :k]
    zero_matrix = np.zeros(shape=matrix.shape)
    for i in range(0, matrix.shape[0]):
        zero_matrix[i, indexes[i]] = matrix[i, indexes[i]]

    return zero_matrix


def compute_pairwise_similarities(matrix, k):
    """ Sets k closest points to 1, rest to 0 """
    indexes = np.argpartition(matrix, k+1, axis=1)[:, :k+1]
    zero_matrix = np.zeros(shape=matrix.shape)
    for i in range(0, matrix.shape[0]):
        zero_matrix[i, indexes[i]] = np.ones((1, k+1))
    zero_matrix -= np.identity(zero_matrix.shape[0])
    return zero_matrix
