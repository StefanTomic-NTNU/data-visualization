import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """ hello this is class docstring"""
    def __init__(self, filename):
        """ hello this is init docstring method"""
        self.col = 0
        self.D = 0
        if filename == 'swiss_data.csv':
            self.col = 2000
            self.D = 3
            self.choice = 0
        elif filename == 'digits.csv':
            self.col = 5620
            self.D = 64
            self.choice = 1
            self.labels = np.genfromtxt('./csv/digits_label.csv', delimiter=',').tolist()

        if self.col == 0 and self.D == 0:
            print("wrong filename")
        else:
            X = np.genfromtxt('./csv/' + filename, delimiter=',')
            self.x = np.reshape(X, (self.col, self.D))
            self.F = 0

    def fit(self):
        """ hello this is docstring for fit method"""
        self.x = self.x - np.mean(self.x, axis=0)
        covmatrix = np.cov(self.x.T)

        [eigenvalues, eigenvectors] = np.linalg.eig(covmatrix)

        eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        self.F = np.hstack((eig_pairs[0][1].reshape(self.D,1), eig_pairs[1][1].reshape(self.D,1)))

    def transform(self):
        """ hello this is docstring for transform method"""
        y = self.F.T @ (self.x.T)
        if self.choice == 0 :
            plt.scatter(y[1,:], y[0,:], c=np.arange(self.col), cmap='gist_rainbow', s=10)
        elif self.choice == 1:
            plt.scatter(y[0,:], y[1,:], c=self.labels, cmap='jet', s=10, marker=".")
        plt.show()

def main():
    """main function"""
    filename = "swiss_data.csv"
    pca = PCA(filename)
    pca.fit()
    pca.transform()



if __name__ == "__main__":
    main()