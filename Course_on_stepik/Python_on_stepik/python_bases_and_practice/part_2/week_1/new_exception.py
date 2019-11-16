class NonPositiveError(Exception):
    pass

class PositiveList(list):
    def append(self, num):
        if num > 0:
            x = super(PositiveList, self).append(num)
            return x
        raise NonPositiveError
            

pl = PositiveList([1, 3])
x = 3
z = pl.append(x)
print(z)