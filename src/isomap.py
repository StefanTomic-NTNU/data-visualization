""" Contains Isomap class """

# TODO: REMOVE SYS BEFORE DELIVERY
import sys
import numpy as np
import matplotlib.pyplot as plot
import scipy as sp
from numpy.linalg import matrix_power
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
        the 50 smallest values.
        """
        # Calculating euclidean distances
        print("Calculating euclidean distances.. ")
        dist_digs = u.calculate_euclidean_distances(self.raw_dig)
        dist_sw = u.calculate_euclidean_distances(self.raw_sw)

        # Reduce matrix to approximate distances along manifold
        red_digs = u.reduce_matrix(dist_digs, 50)
        red_sw = u.reduce_matrix(dist_sw, 50)

        # print("Printing red_sw")
        # with np.printoptions(threshold=sys.maxsize):
        #     print(red_sw)

        # print("^red_sw")

        # print(red_digs)
        # print(red_sw)
        # print(red_digs.shape)
        # print(red_sw.shape)

        print("Computing shortest path.. This may take some time (about 3-5 minutes)\n")
        self.shortest_dig = graph_shortest_path(red_digs)
        self.shortest_sw = graph_shortest_path(red_sw)
        print("Shortest paths computed!\n ")

        # with np.printoptions(threshold=sys.maxsize):
        #     print(self.shortest_sw)

        print(self.shortest_sw)
        print("^Shortest paths (sw)")

        # print(self.shortest_dig)
        # print(self.shortest_sw)
        # print(self.shortest_dig.shape)
        # print(self.shortest_sw.shape)

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
        # dig_mean = np.full((dig_dim, dig_dim), np.mean(squared_dig))
        # sw_mean = np.full((sw_dim, sw_dim), np.mean(squared_sw))
        dig_n = np.full((dig_dim, dig_dim), 1/dig_dim)
        sw_n = np.full((sw_dim, sw_dim), 1/sw_dim)

        # print("\nDIG ID: ")
        # print(dig_id)
        # print("SW ID:")
        # print(sw_id)

        # print("\nDIG MEAN: ")
        # print(dig_mean)
        # print("SW MEAN: ")
        # print(sw_mean)

        cent_mat_dig = np.subtract(dig_id, dig_n)
        cent_mat_sw = np.subtract(sw_id, sw_n)

        print("\nCENT DIG:")
        print(cent_mat_dig)
        print("CENT SW: ")
        print(cent_mat_sw)

        b_dig = 0.5 * np.matmul(cent_mat_dig, squared_dig, dtype=float)
        b_sw = 0.5 * np.matmul(cent_mat_sw, squared_sw, dtype=float)

        print("\nB DIG:")
        print(b_dig)
        print("B SW: ")
        print(b_sw)

        # Eigendecomposition
        print("Performing eigendecomposition.. \n")
        eig_dig = eigh(b_dig, subset_by_index=[0, 1])
        eig_sw = eigh(b_sw, subset_by_index=[0, 1])
        print("Dig Eigens: ")
        print(eig_dig)
        print()
        print("SW Eigens: ")
        print(eig_sw)
        print()

        print(eig_dig[0][0])
        print(eig_dig[0][1])
        sqrt_dig_0 = eig_dig[0][0]
        sqrt_dig_1 = eig_dig[0][1]
        exp_lambda_dig = np.array([[sqrt_dig_0, 0],
                                   [0, sqrt_dig_1]])

        sqrt_sw_0 = eig_sw[0][0]
        sqrt_sw_1 = eig_sw[0][1]
        exp_lambda_sw = np.array([[sqrt_sw_0, 0],
                                  [0, sqrt_sw_1]])

        # Mapping points
        print("Mapping points.. ")
        y_dig = np.matmul(eig_dig[1], exp_lambda_dig)
        y_sw = np.matmul(eig_sw[1], exp_lambda_sw)
        print()

        # Plotting mapped points
        print("Plotting points.. ")
        plot.scatter(y_dig[:, 0], y_dig[:, 1], s=10, marker=".")
        plot.show()
        plot.scatter(y_sw[:, 0], y_sw[:, 1], s=10, marker=".")
        plot.show()
