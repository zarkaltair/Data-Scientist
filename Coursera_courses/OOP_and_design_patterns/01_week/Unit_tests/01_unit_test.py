def bubble_sort(A):
    """
    Sort on place with Bubblesort algorithm
    :type A: list
    """
    for bypass in range(1, len(A)):
        for k in range(0, len(A) - bypass):
            if A[k] > A[k+1]:
                A[k], A[k+1] = A[k+1], A[k]

def test_sort():
    A = [4, 2, 5, 1, 3]
    B = [1, 2, 3, 4, 5]
    bubble_sort(A)
    print("#1:", "Ok" if A == B else 'Fail')

    A = list(range(40, 80)) + list(range(40))
    B = list(range(80))
    bubble_sort(A)
    print('#2:', 'Ok' if A == B else 'Fail')

    A = [4, 2, 4, 2, 1]
    B = [1, 2, 2, 4, 4]
    bubble_sort(A)
    print('#3:', 'Ok' if A == B else 'Fail')

print(test_sort())
