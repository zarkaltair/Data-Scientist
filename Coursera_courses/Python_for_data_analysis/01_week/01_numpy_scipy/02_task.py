from scipy import linalg
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


A = np.array([[1, 3, 5], [2, 5, 1], [2, 3, 8]])
print(linalg.det(A))
print(linalg.inv(A))

eigenvalues, eigenvectors = linalg.eig(A)
print(eigenvalues)
print(eigenvectors)


def f(x):
    return x**2 + 6*np.sin(x)

x  = np.arange(-10, 10, 0.1)
plt.plot(x, f(x))
plt.show()

print(optimize.minimize(f, x0=0))
print(optimize.minimize(f, x0=3))
