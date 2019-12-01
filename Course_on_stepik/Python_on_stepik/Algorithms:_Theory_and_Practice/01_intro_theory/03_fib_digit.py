def fib_digit(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b % 10, (a+b) % 10
    return a

print(fib_digit(6765))
