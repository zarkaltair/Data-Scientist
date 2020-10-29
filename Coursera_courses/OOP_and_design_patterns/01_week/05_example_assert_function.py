def fibonacci(n):
    assert n >= 0
    F = [0, 1] + [0]*n
    for i in range(2, n+1):
        F[i] = F[i-1] + F[i-2]
    assert F[n] >= 0 and F[n] == F[n-1] + F[n-2]
    return F[n]
