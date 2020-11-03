import unittest
import sys


def is_not_in_descending_order(a):
    """
    Check if the list a is not descending (means "rather ascending")
    """
    for i in range(len(a)-1):
        if a[i] > a[i+1]:
            return False
    return True


class TestSort(unittest.TestCase):       
    def test_simple_cases(self):
        self.cases = ([1], [], [1, 2], [1, 2, 3, 4, 5], 
                      [4, 2, 5, 1, 3], [5, 4, 4, 5, 5],
                      list(range(1, 10)), list(range(9, 0, -1)))
        for b in self.cases:
            with self.subTest(case=b):
                a = list(b)
                sort_algorithm(a)
                self.assertCountEqual(a, b,
                                      msg="Elements changed. a = "+str(a))
                self.assertTrue(is_not_in_descending_order(a),
                                msg="List not sorted. a = "+str(a))
                
    def test_stability(self):
        self.cases = ([[0] for i in range(5)],
                      [[1, 2], [2, 2], [2, 3], [2, 2], [2, 3], [1, 2]],
                      [[5, 2], [10, 5], [5, 2], [10, 5], [5, 2], [10, 5]])
    
        for b in self.cases:
            with self.subTest(case=b):
                a = list(b)
                sort_algorithm(a)
                b.sort()  # here we are cheating: standard sort is stable
                # to test stability we will check a[i] is b[i]
                self.assertTrue(all(x is y for x, y in zip(a, b)))
    
    def test_universality(self):
        self.cases = ([4, 2, 8], list('abcdefg'),
                      [True, False],
                      [float(i)/10 for i in range(10, 0, -1)],
                      [[1, 2], [2], [3, 4], [3, 4, 5], [6, 7]])
        for b in self.cases:
            with self.subTest(case=b):
                a = list(b)
                sort_algorithm(a)
                self.assertCountEqual(a, b,
                                      msg="Elements changed. a = "+str(a))
                self.assertTrue(is_not_in_descending_order(a),
                                msg="List not sorted. a = "+str(a))


def bubble_sort(A: list):
    """
    Sorting of list in place. Using Bubble Sort algorithm.
    """
    N = len(A)
    list_is_sorted = False
    bypass = 1
    while not list_is_sorted:
        list_is_sorted = True
        for k in range(N - bypass):
            if A[k] > A[k+1]:
                A[k], A[k+1] = A[k+1], A[k]
                list_is_sorted = False
        bypass += 1


def doing_nothing(A: list):
    """
    Doing nothing with the list A.
    """
    pass


def sort_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestSort('test_simple_cases'))
    suite.addTest(TestSort('test_stability'))
    suite.addTest(TestSort('test_universality'))
    return suite


if True:  # __name__ == '__main__':
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)

    for algo in doing_nothing, bubble_sort:
        print('Testing function ', algo.__doc__.strip())
        test_suite = sort_test_suite()
        sort_algorithm = algo
        runner.run(test_suite)
