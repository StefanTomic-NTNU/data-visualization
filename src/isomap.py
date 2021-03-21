""" Contains Isomap class """

import numpy as np
import matplotlib.pyplot as plot
import scipy as sp
from sklearn.utils.graph_shortest_path import graph_shortest_path

from src import util


class Isomap:
    """ Uses Isometric Mapping to transform data """
    def __init__(self):
        self.raw_digits = util.load_csv_to_array("../csv/digits.csv")
        self.raw_digits_label = util.load_csv_to_array("../csv/digits_label.csv")

    def compute_geodesics(self):
        """
        Finds geodesics (shortest distance) between all points,
        traversing the manifold.
        """
        pass

    def apply_mds(self):
        """
        Applies Multidimentional Scaling (MDS).
        To be used over the geodesic distance matrix.
        """
        pass
