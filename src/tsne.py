""" File containing TSNE class """
import numpy as np
import matplotlib as plt

from src import util as u


class TSNE:
    """ Class for performing Student t-Distributed Stochastic Neighbor Embedding """
    def __init__(self, filename):
        self.raw = u.load_csv_to_array(filename)
        self.similarity_matrix = None

    def compute_pairwise_similarities(self, k):
        """
        Computes pairwise similarities between the raw data points.
        Creates a new matrix filled with zeroes, except for at the
        indexes of the k nearest points in the distance matrix, where
        the value is set to 1.

        Sets the similarity_matrix of self to be the resulting matrix.
        """
        distance_matrix = u.calculate_euclidean_distances(self.raw)
        self.similarity_matrix = u.compute_pairwise_similarities(distance_matrix, k)
        print(self.similarity_matrix)
        print(self.similarity_matrix.shape)

    def map_data_points(self):
        """ Maps data points. """
        pass
