for number in range(5):
    print(number & 1)

for letter in 'python':
    print(ord(letter))

iterator = iter([1, 2, 3])

print(next(iterator))
print(next(iterator))
print(next(iterator))
# print(next(iterator))

class SquareIterator:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration

        result = self.current ** 2
        self.current += 1
        return result

for num in SquareIterator(1, 4):
    print(num)


class IndexIterable:
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, index):
        return self.obj[index]


for letter in IndexIterable('str'):
    print(letter)
