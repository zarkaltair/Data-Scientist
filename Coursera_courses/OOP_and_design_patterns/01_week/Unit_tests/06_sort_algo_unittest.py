import unittest


def sort_algorithm(A: list):
    pass  # FIXME


def is_not_in_descending_order(a):
    """
    Check if the list a is not descending (means "rather ascending")
    """
    for i in range(len(a)-1):
        if a[i] > a[i+1]:
            return False
    return True


class TestSort(unittest.TestCase):
    def setUp(self):
        self.cases = ([1], [], [1, 2], [1, 2, 3, 4, 5], 
                      [4, 2, 5, 1, 3], [5, 4, 4, 5, 5],
                      list(range(1, 10)), list(range(9, 0, -1)))
        
    def test_simple_cases(self):
        for b in self.cases:
            with self.subTest(case=b):
                a = list(b)
                sort_algorithm(a)
                self.assertCountEqual(a, b,
                                      msg="Elements changed. a = "+str(a))
                self.assertTrue(is_not_in_descending_order(a),
                                msg="List not sorted. a = "+str(a))
                
    def tearDown(self):
        self.cases = None


if True: #__name__ == "__main__":
    unittest.main()
