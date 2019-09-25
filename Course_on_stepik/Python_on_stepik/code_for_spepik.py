# def modify_list(l):
#     for i in l:
#         if i % 2 != 0:
#             l.remove(i)
#         else:
#             # l.pop(i)
#             l.append(i)
#             # pass

# l = [1,2,3,4,5,6,7,8]
# modify_list(l)
# print(l)


# def modify_list(l):
#     for i in range(len(l) - 1, -1, -1):
#         if l[i] % 2 != 0:
#             print(l[i])
#             del l[i]
#             print(l)
#         else:
#             print(l[i])
#             l[i] = int(l[i] / 2) #int(i /2)
#             print(l)

# l = [1,2,3,4,5,6]
# modify_list(l)
# print(l)


# def modify_list(l):
#     l[:] = [i//2 for i in l if not i % 2]

# l = [1,2,3,4,5,6]
# modify_list(l)
# print(l)


# def modify_list(l):
#     l[:] = [int(i / 2) for i in l if i % 2 == 0]

# l = [1,2,3,4,5,6]
# modify_list(l)
# print(l)


# def update_dictionary(d, key, value):
#     if key in d:
#         d[key].append(value)
#     elif key not in d:
#         key = key * 2
#         if key in d:
#             d[key].append(value)
#         else:
#             d[key] = [value]

# d = {}
# print(update_dictionary(d, 1, -1))  # None
# print(d)                            # {2: [-1]}
# update_dictionary(d, 2, -2)
# print(d)                            # {2: [-1, -2]}
# update_dictionary(d, 1, -3)
# print(d)                            # {2: [-1, -2, -3]}


# s = 'a aa abC aa ac abc bcd a'

# def words_count(s):
#     arr = {i: s.count(i) for i in s.lower().split(' ')}
    
#     return arr

# print(words_count(s))


# arr = {'x': 5}

# if 'x' in arr:
#     print('Yes')
# else:
#     print('No')

# n = int(input())
# z = 0
# arr = {}
# while z < n:
#     z += 1
#     x = int(input())
#     if x not in arr:
#         a = f(x)
#         arr[x] = a
#         print(a)
#     else:
#         print(arr[x])

# a = [int(input()) for i in range(int(input()))]
# b = {x: f(x) for x in set(a)}
# for i in a:
#     print(b[i])


# import sys

# for i in sys.argv:
#     print(i)
    # pass

# print(sys.argv)
# print(dir(sys.argv))
# print(*sys.argv)


# print(__import__("math").pi)

# print(__import__('numpy').pi)


# from itertools import groupby

# with open('dataset_3363_2.txt', 'r') as file:
# # with open('input.txt', 'r') as file:
#     s = file.readline().strip()

# arr = [''.join(i) for _, i in groupby(s, str.isalpha)]
# arr_alpha = [i for i in arr if i.isalpha()]
# arr_digit = [i for i in arr if i.isdigit()]

# # print(len(arr_alpha))
# # print(len(arr_digit))
# ans = ''.join([arr_alpha[i] * int(arr_digit[i]) for i in range(len(arr_alpha))])
# print(ans)
# with open('answer_1.txt', 'w') as file:
#     file.write(ans)


# from collections import Counter

# arr = []
# with open('dataset_3363_3.txt', 'r') as file:
#     for line in file:
#         s = line.strip().split(' ')
#         for i in s:
#             arr.append(i.lower())

# c = Counter(arr).most_common(1)
# ans = ' '.join([str(i) for i in c[0]])
# with open('answer_2.txt', 'w') as file:
#     file.write(ans)


# arr_mean = []
# arr_math, arr_fis, arr_rus = [], [], []
# with open('dataset_3363_4.txt', 'r') as file:
#     for line in file:
#         arr = [int(i) for i in line.strip().split(';') if i.isdigit()]
#         arr_mean.append(str(sum(arr) / len(arr)))
#         arr_math.append(arr[0])
#         arr_fis.append(arr[1])
#         arr_rus.append(arr[2])

# with open('answer_3.txt', 'w') as file:
#     file.write('\n'.join(arr_mean))
#     file.write('\n' + str(sum(arr_math) / len(arr_math)) + ' ')
#     file.write(str(sum(arr_fis) / len(arr_fis)) + ' ')
#     file.write(str(sum(arr_rus) / len(arr_rus)))


# import requests

# with open('dataset_3378_2.txt', 'r') as file:
#     s = file.read().strip()
#     res = requests.get(s).text
#     ans = res.splitlines()

# with open('answer_4.txt', 'w') as file:
#     file.write(str(len(ans)))


import requests

def get_file(url):
    return requests.get(url).text

with open('dataset_3378_3.txt', 'r') as file:
    s = file.read().strip()
    res = get_file(s)
    while 'We' not in res:
        url = 'https://stepic.org/media/attachments/course67/3.6.3/' + res
        res = get_file(url)
        print(res)
        
with open('answer_5.txt', 'w') as file:
	file.write(res)