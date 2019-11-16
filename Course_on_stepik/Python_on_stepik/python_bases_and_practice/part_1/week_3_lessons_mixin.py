class EvenLengthMixin:
    def even_length(self):
        return len(self) % 2 == 0


class MyList(list, EvenLengthMixin):
    def pop(self):
        x = super(MyList, self).pop()
        print('Last value is', x)
        return x


class MyDict(dict, EvenLengthMixin):
    pass


x = MyDict()
x['key'] = 'value'
x['another_key'] = 'another_value'
print(x.even_length()) # True


ml = MyList([1, 2, 4, 17])
z = ml.pop()
print(z)
print(ml)