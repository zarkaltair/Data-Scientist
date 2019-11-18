from random import random


class RandomIterator:
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

# for i in RandomIterator(10):
#     print(i)


def random_generator(k):
    for i in range(k):
        yield random()

gen = random_generator(3)
print(type(gen))
for i in gen:
    print(i)