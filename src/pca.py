import numpy as np
import matplotlib.pyplot as plot
import scipy as sp

class PCA:
    def __init__(self):
        X = np.genfromtxt('./csv/swiss_data.csv', delimiter=',')
        x = np.reshape(X, (2000, 3))
        # x is a vector of p random variables

        N = len(x) #size of all the x in csv file 
        u = 1/N * np.sum(x)

        #sigma is know to be the covariance matrix of x 


        for i in range(N - 1):
            x[i] = x[i] - u
        np.cov(x)


        pass

    def fit(self):

        pass
    
    def transform(self):
        
        pass
