import sys


sys.stdin = open('test.txt', 'r')

def find(s, a):
    return True if a in s else False

s, a, b, x = input(), input(), input(), 0

while find(s, a):
    x += 1
    s = s.replace(a, b)
    if x > 1000:
        break

if x > 1000:
    print('Impossible')
else:
    print(x)