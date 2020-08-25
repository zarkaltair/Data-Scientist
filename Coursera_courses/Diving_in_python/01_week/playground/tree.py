import sys


n = int(sys.argv[1])

for i in range(1, n+1):
    print(' ' * (n - i) + '#' * i)
