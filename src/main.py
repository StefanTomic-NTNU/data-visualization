""" Contains main function of project """
from src.isomap import Isomap
from src.pca import PCA


def main():
    """ Main function of project """
    pca1 = PCA()
    isomap1 = Isomap()
    isomap1.compute_geodesics()
    isomap1.apply_mds()


if __name__ == '__main__':
    main()
