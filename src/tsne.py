""" File containing TSNE class """
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

from src import util as u


class TSNE:
    """ Class for performing Student t-Distributed Stochastic Neighbor Embedding """
    def __init__(self, filename):
        self.filename = filename
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

    def map_data_points(self, max_iteration, alpha, epsilon):
        """ Maps data points. """

        # divide each point by sum of values
        stand_hd_similarity_matrix = self.hd_similarity_matrix / np.sum(self.hd_similarity_matrix)  # P
        dynamic_stand_hd_similarity_matrix = 4 * stand_hd_similarity_matrix

        # Sample 2D data points from normal distribution
        # TODO: This should be done by seed when algorithm is somewhat correct.
        sampled_two_d_points = normal(0, 10e-4, (2, self.nr_data_points))

        # Initialize variables
        gain = np.ones((2, self.nr_data_points))        # g in assignment
        change = np.zeros((2, self.nr_data_points))     # delta in assignment
        dynamic_alpha = 0.5

        for i in range(1, max_iteration):
            print("Iteration " + str(i))

            if i == 250:
                dynamic_alpha = alpha   # Optimisation trick

            # Find similarity matrix of 2D points
            two_d_similarity_matrix = u.calculate_euclidean_distances(np.swapaxes(sampled_two_d_points, 0, 1))
            two_d_similarity_matrix = 1 / (1 + np.square(two_d_similarity_matrix))  # q as described in assignment

            # divide each point by sum of values
            stand_two_d_similarity_matrix = two_d_similarity_matrix / np.sum(two_d_similarity_matrix)  # Q

            if i == 100:
                dynamic_stand_hd_similarity_matrix = stand_hd_similarity_matrix     # Optimisation trick

            # print("Before")
            # print(sampled_two_d_points)

            # Calculate the gradient over each y_i
            # Gradient is top-down delta in assignment
            # TODO: This is an attempt at implementing 6.2.2a, but I think it is incorrect.
            #  It is done differently in slides.
            # gradient = 4 * ((stand_hd_similarity_matrix.sum(axis=0) - stand_two_d_similarity_matrix.sum(axis=0)) *
            #                 two_d_similarity_matrix.sum(axis=0) *
            #                 (sampled_two_d_points * self.nr_data_points - np.sum(sampled_two_d_points, axis=0)))

            Y = np.swapaxes(sampled_two_d_points, 0, 1)
            G = (dynamic_stand_hd_similarity_matrix - stand_two_d_similarity_matrix) * two_d_similarity_matrix
            S = np.diag(np.sum(G, axis=1))
            gradient = 4 * (S - G) @ Y

            # print(gradient.shape)
            gradient = np.swapaxes(gradient, 0, 1)

            # Update gain
            gain[np.sign(gradient) != np.sign(change)] += 0.2
            gain[np.sign(gradient) == np.sign(change)] *= 0.8
            gain[gain < 0.01] = 0.01

            # Update change
            change = dynamic_alpha * change - epsilon * gain * gradient

            # Update 2D points
            sampled_two_d_points += change

            print(sampled_two_d_points)

        # Plotting mapped points
        if self.filename == "digits.csv":
            labels = u.load_csv_to_array("digits_label.csv").tolist()
            points_to_plot = np.swapaxes(sampled_two_d_points, 0, 1)
            plt.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c=labels, cmap='tab10', s=8, marker=".")
            cbar = plt.colorbar()
            cbar.set_label("Number labels")
        else:
            plt.scatter(sampled_two_d_points[:, 0], sampled_two_d_points[:, 1], s=10, marker=".")
        plt.show()
