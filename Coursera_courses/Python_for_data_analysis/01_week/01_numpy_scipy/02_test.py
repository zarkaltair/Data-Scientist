import numpy as np
from scipy import linalg


a = np.array([[1, 2, 3], [4, 3, 6]])
# print(np.full((2, 2), 5))
# print(np.randint())
b = a.flatten()
# print(b)
# print(a)
# print(a is b)
# print(np.eye(5))
# print(np.random.randint((3, 5, 3)))

Z = np.array([92, 13, 44, 555, 1, -3])[::-1]
# print(Z[4])

a = np.array([1, 1, 1, 1, 1])
b = np.array([2, 2, 2, 2, 2])
# print(np.dot(a, b))

c = np.array([
    [0, 9, 19, 13],
    [1, 20, 5, 13],
    [12, 11, 3, 4]])
d = np.array([
    [2, 0, 0, 0],
    [1, 2, 2, 0],
    [2, 1, 1, 0],
    [0, 0, 1, 1]])

# print(c.dot(d))

b = np.array([[-1, 33, 4, 1], [0, 1, 1, 0]])
# print(np.mean(b))

e = np.array([[6, 0, 3], [0, -1, 2], [12, 3, 0]])
D = linalg.det(e)
# print(D)

m = np.array([
    [1, -1, -1, 0],
    [-1, 2, -1, -1],
    [-1, -1, 2, -1],
    [0, -1, -1, 1]])

wb, vb = linalg.eig(m)
# print(wb)

n = np.array([
    [2, 4, 0, 4, 1],
    [2, 4, 1, 1, 0],
    [1, 1, 1, 2, 2],
    [0, 1, 3, 2, 4],
    [2, 2, 2, 0, 2]])
n_inv = linalg.inv(n)
print(n_inv)