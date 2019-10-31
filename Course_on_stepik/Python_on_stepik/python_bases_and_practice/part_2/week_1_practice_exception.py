import sys

sys.stdin = open('test.txt', 'r')

def recursion(ex):
    if dct[ex] == []:
        ex = ex
    else:
        recursion(dct[ex][0])
    return ex

dct = {}
n = int(input())
for i in range(n):
    conn = input().split(' : ')
    if len(conn) > 1:
        dct[conn[0]] = conn[1].split(' ')
        for i in dct[conn[0]]:
            if i not in dct:
                dct[i] = []
    else:
        dct[conn[0]] = []
s = int(input())
arr = []
for i in range(s):
    arr.append(input())
for i in arr:
    if arr.index(i) != 0:
        ex = recursion(i)
        print(ex)
        if ex in arr[:arr.index(i)]:
            print(ex)

# print(dct)
# print(arr)