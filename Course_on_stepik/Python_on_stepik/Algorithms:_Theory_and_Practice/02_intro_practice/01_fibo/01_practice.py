def fib1(n):
    assert n >= 0
    return n if n <= 1 else fib1(n - 1) + fib1(n - 2)


print(fib1(8))

from rcviz import viz

old_fib1 = fib1

fib1 = viz(fib1)

# fib1(5)

cache = {}

def fib2(n):
    assert n >= 0
    if n not in cache:
        cache[n] = n if n <= 1 else fib2(n - 1) + fib2(n - 2)
    return cache[n]

print(fib2(8))
print(fib2(80))
print(fib2(800))

cache.clear()

fib2 = viz(fib2)
# fib2(5)


def memo(f):
    cache = {}
    def inner(n):
        if n not in cache:
            cache[n] = f(n)
        return cache[n]
    return inner

fib1 = memo(old_fib1)

print(fib1(80))

from functools import lru_cache

fib1 = lru_cache(maxsize=None)(old_fib1)

print(fib1(80))
# print(fib1(8000))

def fib3(n):
    assert n >= 0
    f0, f1 = 0, 1
    for i in range(n - 1):
        f0, f1 = f1, f0 + f1
    return f1

print(fib3(8))
print(fib3(8000))







