""" File containing TSNE class """
import numpy as np
from numpy.random import normal
import matplotlib as plt

from src import util as u


class TSNE:
    """ Class for performing Student t-Distributed Stochastic Neighbor Embedding """
    def __init__(self, filename):
        self.raw = u.load_csv_to_array(filename)
        self.nr_data_points = self.raw.shape[0]
        self.hd_similarity_matrix = None    # p as described in assignment

    def compute_pairwise_similarities(self, k):
        """
        Computes pairwise similarities between the raw data points.
        Creates a new matrix filled with zeroes, except for at the
        indexes of the k+1 nearest points in the distance matrix, where
        the value is set to 1. Diagonal is then set to zero
        (in other words distance from point to itself is not included).

        Sets the hd_similarity_matrix of self to be the resulting matrix.
        """
        distance_matrix = u.calculate_euclidean_distances(self.raw)
        self.hd_similarity_matrix = u.compute_pairwise_similarities(distance_matrix, k)
        # print(self.hd_similarity_matrix)
        # print(self.hd_similarity_matrix.shape)

    def map_data_points(self, max_iteration, alpha, epsilon):
        """ Maps data points. """
        # Sample 2D data points from normal distribution
        sampled_two_d_points = normal(0, 10e-4, (2, self.nr_data_points))

        # Find similarity matrix of 2D points
        sampled_two_d_points2 = np.swapaxes(sampled_two_d_points, 0, 1)
        two_d_similarity_matrix = u.calculate_euclidean_distances(sampled_two_d_points2)
        two_d_similarity_matrix = 1 / (1 + np.square(two_d_similarity_matrix))  # q as described in assignment

        # divide each point by sum of values
        stand_two_d_similarity_matrix = two_d_similarity_matrix / np.sum(two_d_similarity_matrix)   # Q
        stand_hd_similarity_matrix = self.hd_similarity_matrix / np.sum(self.hd_similarity_matrix)  # P

        # Initialize variables
        gain = np.ones((1, self.nr_data_points))        # g in assignment
        change = np.zeros((1, self.nr_data_points))     # delta in assignment

        for i in range(1, max_iteration):
            # Calculate the gradient over each y_i
            # Gradient is top-down delta in assignment
            gradient = 4 * ((stand_hd_similarity_matrix.sum(axis=0) - stand_two_d_similarity_matrix.sum(axis=0)) *
                               two_d_similarity_matrix.sum(axis=0) *
                              (sampled_two_d_points * self.nr_data_points - np.sum(sampled_two_d_points, axis=0)))