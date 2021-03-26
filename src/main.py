""" Contains main function of project """
from isomap import Isomap
from pca import PCA


def main():
    """ Main function of project """
    pca = PCA()
    isomap = Isomap()
    isomap.compute_geodesics()
    isomap.apply_mds()

if __name__ == '__main__':
    main()
