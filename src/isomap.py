""" Contains Isomap class """

import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import multi_dot
from scipy.linalg import eigh
from sklearn.utils.graph_shortest_path import graph_shortest_path

from src import util as u


class Isomap:
    """ Uses Isometric Mapping to transform data """
    def __init__(self, filename):
        """ Loads data from csv file to array """
        self.filename = filename
        self.raw = u.load_csv_to_array(filename)
        self.nr_data_points = self.raw.shape[0]
        self.geodesics = None

    def compute_geodesics(self, k):
        """
        Finds geodesics (shortest distance) between all points,
        traversing the manifold. This is done by first calculating
        euclidean distances, then keeping only the smallest ones.
        This gives an approximated geodesic distance matrix.
        """
        # Calculating euclidean distances
        print("\nCalculating euclidean distances.. ")
        distance_matrix = u.calculate_euclidean_distances(self.raw)

        # Reduce matrix to approximate distances along manifold
        reduced_distance_matrix = u.reduce_matrix(distance_matrix, k)

        # Applying dijkstra's algorithm
        print("Computing shortest path.. ")
        self.geodesics = graph_shortest_path(reduced_distance_matrix)

    def apply_mds(self):
        """
        Applies Multidimentional Scaling (MDS).
        To be used over the geodesic distance matrix.
        """
        # Squaring matrices
        print("\nSquaring matrices.. ")
        squared_geodesics = np.square(self.geodesics)

        # Apply double centering
        print("Double centering.. ")
        centering_matrix = np.subtract(np.identity(self.nr_data_points),
                                       np.full((self.nr_data_points, self.nr_data_points), 1/self.nr_data_points))

        centered_matrix = multi_dot([centering_matrix, squared_geodesics, centering_matrix]) * -0.5

        # Eigendecomposition
        print("Performing eigendecomposition.. ")
        eigens = eigh(centered_matrix,
                      subset_by_index=[self.nr_data_points-2, self.nr_data_points-1])

        squared_lambda = np.array([[np.sqrt(eigens[0][0]), 0],
                                   [0, np.sqrt(eigens[0][1])]])

        # Mapping points
        print("Mapping points to 2D.. ")
        y_dig = eigens[1].dot(squared_lambda)
        y_sw = eigens[1].dot(squared_lambda)

        # Plotting mapped points
        print("Plotting points.. ")
        if self.filename == "swiss_data.csv":
            plot.scatter(y_sw[:, 1], y_sw[:, 0],
                         c=np.arange(self.nr_data_points), cmap='gist_rainbow', s=20, marker=".")
        elif self.filename == "digits.csv":
            labels = u.load_csv_to_array("digits_label.csv").tolist()
            plot.scatter(y_dig[:, 0], y_dig[:, 1], c=labels, cmap='tab10', s=10, marker=".")
            cbar = plot.colorbar()
            cbar.set_label("Number labels")
        else:
            plot.scatter(y_dig[:, 0], y_dig[:, 1], s=10, marker=".")
        plot.show()
