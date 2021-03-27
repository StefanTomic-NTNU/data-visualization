""" Contains Isomap class """

# TODO: REMOVE SYS BEFORE DELIVERY
import sys
import numpy as np
import matplotlib.pyplot as plot
import scipy as sp
from numpy.linalg import matrix_power
from numpy.linalg import multi_dot
from scipy.linalg import eigh
from sklearn.utils.graph_shortest_path import graph_shortest_path

from src import util as u


class Isomap:
    """ Uses Isometric Mapping to transform data """
    def __init__(self):
        """ Loads data from csv files to array """
        print("Isomap initializing.. ")
        print("Loading csv to array.. ")
        self.raw_dig = u.load_csv_to_array("digits.csv")
        self.raw_dig_lab = u.load_csv_to_array("digits_label.csv")
        self.raw_sw = u.load_csv_to_array("swiss_data.csv")
        self.shortest_dig = None
        self.shortest_sw = None
        print("Isomap initialized! \n")

    def compute_geodesics(self):
        """
        Finds geodesics (shortest distance) between all points,
        traversing the manifold. Reduces the vectors to only include
        the k smallest values.
        """
        # Calculating euclidean distances
        print("Calculating euclidean distances.. ")
        dist_digs = u.calculate_euclidean_distances(self.raw_dig)
        dist_sw = u.calculate_euclidean_distances(self.raw_sw)

        # Reduce matrix to approximate distances along manifold
        red_digs = u.reduce_matrix(dist_digs, 50)
        red_sw = u.reduce_matrix(dist_sw, 20)

        # Applying dijkstra's algorithm
        print("Computing shortest path.. This may take some time.. \n")
        self.shortest_dig = graph_shortest_path(red_digs)
        self.shortest_sw = graph_shortest_path(red_sw)

    def apply_mds(self):
        """
        Applies Multidimentional Scaling (MDS).
        To be used over the geodesic distance matrix.
        """
        # Squaring matrices
        print("Squaring matrices..")
        squared_dig = np.square(self.shortest_dig)
        squared_sw = np.square(self.shortest_sw)
        print("SW SQUARED: ")
        print(squared_sw)

        # Apply double centering
        print("Double centering..")
        dig_dim = squared_dig.shape[0]
        sw_dim = squared_sw.shape[0]
        dig_id = np.identity(dig_dim)
        sw_id = np.identity(sw_dim)
        dig_n = np.full((dig_dim, dig_dim), 1/dig_dim)
        sw_n = np.full((sw_dim, sw_dim), 1/sw_dim)

        cent_mat_dig = np.subtract(dig_id, dig_n)
        cent_mat_sw = np.subtract(sw_id, sw_n)

        b_dig = multi_dot([cent_mat_dig, squared_dig, cent_mat_dig])
        b_sw = multi_dot([cent_mat_sw, squared_sw, cent_mat_sw])

        b_dig = b_dig * -0.5
        b_sw = b_sw * -0.5

        # Eigendecomposition
        print("\nPerforming eigendecomposition..")
        eig_dig = eigh(b_dig, subset_by_index=[5618, 5619])
        eig_sw = eigh(b_sw, subset_by_index=[1998, 1999])

        sqrt_dig_0 = np.sqrt(eig_dig[0][0])
        sqrt_dig_1 = np.sqrt(eig_dig[0][1])
        exp_lambda_dig = np.array([[sqrt_dig_0, 0],
                                   [0, sqrt_dig_1]])

        sqrt_sw_0 = np.sqrt(eig_sw[0][0])
        sqrt_sw_1 = np.sqrt(eig_sw[0][1])
        exp_lambda_sw = np.array([[sqrt_sw_0, 0],
                                  [0, sqrt_sw_1]])

        # Mapping points
        print("\nMapping points.. ")
        y_dig = eig_dig[1].dot(exp_lambda_dig)
        y_sw = eig_sw[1].dot(exp_lambda_sw)

        # Plotting mapped points
        print("\nPlotting points.. ")

        plot.scatter(y_dig[:, 0], y_dig[:, 1], s=10, marker=".")
        plot.show()
        plot.scatter(y_sw[:, 1], y_sw[:, 0], c=np.arange(2000), s=10, marker=".")
        plot.show()

