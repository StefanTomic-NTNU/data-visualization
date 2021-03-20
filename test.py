import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

X = np.genfromtxt('./csv/swiss_data.csv', delimiter=',')
x = np.reshape(X, (2000, 3))
x = x.T
D = 3
print(x)
u = (1/2000) * np.sum(x)

for i in range(3):
    for j in range(2000):
        x[i][j] = x[i][j] - u
covmatrix = np.cov(x)


d = 2
[eigenvalues, eigenvectors] = np.linalg.eig(covmatrix)
print("z")
print(eigenvectors)
print("x")
print(eigenvalues)
F = eigenvectors

for i in range(len(eigenvalues)):
    eigvec_cov = eigenvectors[:,i].reshape(1,3).T
print("jup")
print(eigvec_cov)
eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

for i in eig_pairs:
    print(i[0])


F = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))

y = F.T.dot(x-u)

print(y)
plt.plot(y[0,0:1000], y[1,0:1000])
plt.plot(y[0,1000:2000], y[1,1000:2000])
plt.show()


