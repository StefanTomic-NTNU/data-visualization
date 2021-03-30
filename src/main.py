""" Contains main function of project """
from src.isomap import Isomap
from src.pca import PCA


def main():
    """ Main function of project """
    # PCA
    pca = PCA("swiss_data.csv")
    pca.fit()
    pca.transform()

    # Isomap
    # isomap_digits = Isomap("digits.csv")
    # isomap_swiss = Isomap("swiss_data.csv")
    # isomap_digits.compute_geodesics(35)     # Value of 30-40 seems to be best fit
    # isomap_swiss.compute_geodesics(25)      # Value of 20-30 seems to be best fit
    # isomap_digits.apply_mds()
    # isomap_swiss.apply_mds()


if __name__ == '__main__':
    main()
