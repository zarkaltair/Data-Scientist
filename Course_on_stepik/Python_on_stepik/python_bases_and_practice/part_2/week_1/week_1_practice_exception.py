import sys
from pprint import pprint


sys.stdin = open('test_4.txt', 'r')

exceptions = {}
throwed_exceptions = []

def found_path(exceptions, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if start not in exceptions:
        return []
    for node in exceptions[start]:
        if node not in path:
            newpath = found_path(exceptions, node, end, path)
            if newpath:
                return newpath
    return []

n = int(input())
for _ in range(n):
    inpt = input().split()
    child = inpt[0]
    parents = inpt[2:]
    exceptions[child] = parents

m = int(input())
for _ in range(m):
    throwing = input()
    for throwed_exception in throwed_exceptions:
        if len(found_path(exceptions, throwing, throwed_exception)) > 1:
            print(throwing)
            break
    throwed_exceptions.append(throwing)

'''
def recursion(ex):
    if dct[ex] != []:
        ex = dct[ex][0]
        ex = recursion(ex)
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
arr_ex = []
for i in range(s):
    arr.append(input())
for i in arr:
    if arr.index(i) != 0:
        ex = recursion(i)
        if ex in arr[:arr.index(i)] or ex in arr_ex:
            print(i)
        else:
            arr_ex.append(ex)

# pprint(dct)
# print(arr)
# print(arr_ex)


dd = {'1': [], 
      '2': ['1'], 
      '3': ['1', '7', '8', '2'], 
      '4': ['1', '6'], 
      '5': ['1', '7', '2', '3'], 
      '6': ['1', '9'], 
      '7': ['2'], 
      '8': ['7'], 
      '9': ['8', '2', '3', '5'], 
      '10': ['6']}

arr = ['5', '9', '1', '7', '8']

ans = '1 7 8'


dd = {'A': [], 
      'H': [], 
      'G': [],
      'DH': [], 
      'BG': [], 
      'B': ['E'], 
      'CE': ['C'], 
      'C': ['C', 'G'], 
      'D': ['D', 'H'], 
      'E': ['C', 'H'], 
      'F': ['B', 'F'], 
      'BF': ['D', 'H'], 
      'DE': ['D', 'H'], 
      'CH': ['B', 'G'], 
      'BH': ['D', 'E'], 
      'AE': ['D', 'H', 'D'], 
      'CG': ['E', 'B', 'H']}

arr = ['DH', 'D', 'B', 'AE', 'BG', 'H', 'E', 'BH', 'CE', 'CH', 'C', 'A', 'DE', 'F', 'CG', 'G', 'BF']

ans = ['BG', 'E', 'CH', 'C', 'DE', 'CG', 'BF']


dd = {'1': ['2', '3', '4', '5', '6'], 
      '2': ['7', '3', '5', '10', '9'], 
      '3': ['5', '9'], 
      '4': ['10'], 
      '5': ['9'], 
      '6': ['4', '10'], 
      '7': ['8', '3', '5'], 
      '8': ['3', '9'], 
      '9': ['6'], 
      '10': []}

arr = ['5', '9', '1', '7', '8']

ans = '1 7 8'


dd = {'a': [], 
      'b': ['a'], 
      'c': ['a'], 
      'f': ['a'], 
      'd': ['c', 'b'], 
      'g': ['d', 'f'], 
      'i': ['g'], 
      'm': ['i'], 
      'n': ['i'], 
      'z': ['i'], 
      'e': ['m', 'n'], 
      'y': ['z'], 
      'x': ['z'], 
      'w': ['e', 'y', 'x']}

arr = ['y', 'm', 'n', 'm', 'd', 'e', 'g', 'a', 'f']

ans = 'm e g f'
'''


'''
def recursion(ex):
    if dct[ex] != []:
        ex = dct[ex][0]
        ex = recursion(ex)
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
arr_ex = []
for i in range(s):
    arr.append(input())
for i in arr:
    if arr.index(i) != 0:
        ex = recursion(i)
        arr_ex.append(ex)
        if ex in arr[:arr.index(i)] or ex in arr_ex:
            print(i)

dct = {}
n = int(input())
for i in range(n):
    conn = input().split(' : ')
    if len(conn) > 1:
        if conn[0] not in dct:
            dct[conn[0]] = []
        for i in conn[1].split():
            if i not in dct:
                dct[i] = []
                dct[i] += conn[0]
            else:
                dct[i] += conn[0]
    else:
        dct[conn[0]] = []
'''