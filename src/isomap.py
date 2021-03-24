""" Contains Isomap class """

import numpy as np
import matplotlib.pyplot as plot
import scipy as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.graph_shortest_path import graph_shortest_path

from src import util as u


class Isomap:
    """ Uses Isometric Mapping to transform data """
    def __init__(self):
        """ Loads data from csv files to array """
        self.raw_dig = u.load_csv_to_array("digits.csv")
        self.raw_dig_lab = u.load_csv_to_array("digits_label.csv")
        self.raw_sw = u.load_csv_to_array("swiss_data.csv")
        self.shortest_dig = None
        self.shortest_sw = None

    def compute_geodesics(self):
        """
        Finds geodesics (shortest distance) between all points,
        traversing the manifold. Reduces the vectors to only include
        the 50 smallest values.
        """

        dist_digs = u.calculate_euclidean_distances(self.raw_dig)
        dist_sw = u.calculate_euclidean_distances(self.raw_sw)

        red_digs = u.reduce_matrix(dist_digs, 50)
        red_sw = u.reduce_matrix(dist_sw, 50)

        print(red_digs)
        print(red_sw)
        print(red_digs.shape)
        print(red_sw.shape)

        self.shortest_dig = graph_shortest_path(red_digs)
        self.shortest_sw = graph_shortest_path(red_sw)

        print(self.shortest_dig)
        print(self.shortest_sw)
        print(self.shortest_dig.shape)
        print(self.shortest_sw.shape)


    def apply_mds(self):
        """
        Applies Multidimentional Scaling (MDS).
        To be used over the geodesic distance matrix.
        """
        pass
