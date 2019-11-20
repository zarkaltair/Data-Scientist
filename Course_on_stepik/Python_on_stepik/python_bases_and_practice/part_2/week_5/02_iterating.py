x = input().split()
xs = (int(i) for i in x)

# def even(x):
#     return x % 2 == 0

# evens = filter(even, xs)
# for i in evens:
#     print(i)


evens = list(filter(lambda x: x % 2 == 0, xs))
print(evens)