import numpy as np


x = np.array([1, 2, 3, 4], dtype=np.int8)
print(x.dtype)

print(x.shape)

m = np.array([[2, 3, 4], [5, 6, 7]])
print(m.shape)

print(np.ones(5))
print(np.zeros(5))
print(np.eye(6))
print(np.random.random((2, 3)))

print(x[1])
print(x[0])

y = np.array([[1, 3, 4, 4], [2, 3, 4, 56]])
print(y)
print(y[1, 3])
print(y[:, :3])
print(y > 2)
print(y[y > 2])

b = np.array([[1, 3, 4, 4, 5], [2, 3, 4, 56, 5]])
print(b.shape)
print(b.flatten())
print(b.T)
print(b.reshape((10, 1)))
b.resize((10, 1))
print(b)

v = np.array([9, 10])
w = np.array([11, 12])
print(v + w)
print(np.add(v, w))
print(v - w)
print(np.subtract(v, w))
print(v * w)
print(np.multiply(v, w))
print(np.dot(v, w))
