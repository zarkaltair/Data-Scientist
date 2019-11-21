from random import random


class RandomIterator:
    """
    RandomIterator(k) - new iterator for k random number in [0, 1)

    Uses random.random
    """
    def __iter__(self):
        return self

    def __init__(self, k):
        self.k = k
        self.i = 0

    def __next__(self):
        if self.i < self.k:
            self.i += 1
            return random()
        else:
            raise StopIteration

print(RandomIterator.__doc__)

print(random.__doc__)