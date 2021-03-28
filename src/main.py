""" Contains main function of project """
from src.isomap import Isomap
from tsne import TSNE
from src.pca import PCA


def main():
    """ Main function of project """
    # PCA
    # pca_digits = PCA()

    # Isomap
    # isomap_digits = Isomap("digits.csv")
    # isomap_swiss = Isomap("swiss_data.csv")
    # isomap_digits.compute_geodesics(35)     # Value of 30-40 seems to be best fit
    # isomap_swiss.compute_geodesics(25)      # Value of 20-30 seems to be best fit
    # isomap_digits.apply_mds()
    # isomap_swiss.apply_mds()

    # t-SNE
    tsne_digits = TSNE("digits.csv")
    tsne_digits.compute_pairwise_similarities(40)
    # Values given in assignment are:
    # max_iteration:    1000
    # alpha:            0.8
    # epsilon:          500
    tsne_digits.map_data_points(50, 0.8, 500)


if __name__ == '__main__':
    main()
