x = input().split()
# ['100', '500']
print(x)
map_obj = map(int, x)
# n, k = (int() for i in x)
# <map object at 0x7fd81867c4d0>
print(map_obj)
n = next(map_obj)
k = next(map_obj)
# 600
print(n + k)