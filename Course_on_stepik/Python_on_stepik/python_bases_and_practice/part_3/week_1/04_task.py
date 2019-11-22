import sys


sys.stdin = open('test_2.txt', 'r')

s, t, x = input(), input(), 0

for i in range(len(s)):
    if s[i:].startswith(t):
        x += 1

print(x)