""" Contains Isomap class """

import numpy as np
import matplotlib.pyplot as plot
import scipy as sp
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

        # print(red_digs)
        # print(red_sw)
        # print(red_digs.shape)
        # print(red_sw.shape)

        print("Computing shortest path.. This may take some time (about 3-5 minutes)\n")
        self.shortest_dig = graph_shortest_path(red_digs)
        self.shortest_sw = graph_shortest_path(red_sw)
        print("Shortest paths computed!\n ")

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
        print()

        # Apply double centering
        print("Double centering..")
        dig_dim = squared_dig.shape[0]
        sw_dim = squared_sw.shape[0]
        cent_mat_dig = np.identity(dig_dim) - np.full((dig_dim, dig_dim), np.mean(squared_dig))
        cent_mat_sw = np.identity(sw_dim) - np.full((sw_dim, sw_dim), np.mean(squared_sw))
        b_dig = ((-1)/2) * cent_mat_dig * squared_dig
        b_sw = ((-1)/2) * cent_mat_sw * squared_sw
        print()

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

        # Mapping points
        print("Mapping points.. ")
