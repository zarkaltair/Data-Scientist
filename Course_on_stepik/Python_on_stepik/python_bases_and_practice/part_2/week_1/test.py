import sys


sys.stdin = open('test_2.txt', 'r')

def recursion_bypass(class_1, class_2):
    s = []
    if class_1 == class_2:
        s.append('Yes')
    else:
        arr = dct[class_2]
        if class_1 in arr:
            s.append('Yes')
        elif len(arr) >= 1:
            for i in arr:
                s.append(recursion_bypass(class_1, i))
        else:
            s.append('No')
    return 'Yes' if 'Yes' in s else 'No'

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
print(dct)
'''
s = int(input())
for i in range(s):
    quest = input().split(' ')
    print(recursion_bypass(quest[0], quest[1]))
'''