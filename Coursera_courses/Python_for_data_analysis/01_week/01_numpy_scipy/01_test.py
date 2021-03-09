import numpy as np


a = np.matrix('2 -1 0; 0 1 3; 4 -3 -3')
print(a)

d = np.linalg.det(a)
print(d)

b = np.matrix('1 13; 2 3')
print(b)

d = np.linalg.det(b)
print(d)

c = np.matrix('42 1 6; 1 0 -3; 3 5 1')
e = np.matrix('-1 1; 0 7; 9 -2')
print(c)
print(e)

total = c.dot(e)
print(total)

m = np.matrix('2 1 -3; 1 0 0; 1 0 1')
m_inv = np.linalg.inv(m)
print(m_inv)

n = np.matrix('0 3; 3 8')
wb, vb = np.linalg.eigh(n)
print(wb)
print(vb)

b = np.matrix('2 3;1 13')
print(b)

d = np.linalg.det(b)
print(d)
