import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class PCA:
    def __init__(self, dimension, filename):
        #add filename here
        X = np.genfromtxt('./csv/swiss_data.csv', delimiter=',')
        self.col = 2000
        x = np.reshape(X, (2000, 3))
        self.x = x.T
        self.D = dimension
        self.F = 0
        self.u = 0
        

    def fit(self, d):
        self.u = (1/self.col) * np.sum(self.x)
        for i in range(self.D):
            for j in range(self.col):
                self.x[i][j] = self.x[i][j] - u
        covmatrix = np.cov(self.x)

        [eigenvalues, eigenvectors] = np.linalg.eig(covmatrix)

        for i in range(len(eigenvalues)):
            eigvec_cov = eigenvectors[:,i].reshape(1,self.D).T

        eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        self.F = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))

    def transform(self):
        y = self.F.T.dot(self.x -self.u)
        plt.plot(y[0,0:1000], y[1,0:1000])
        plt.plot(y[0,1000:2000], y[1,1000:2000])
        plt.show()
