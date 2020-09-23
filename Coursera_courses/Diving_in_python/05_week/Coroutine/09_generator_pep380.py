def chain(x_iterable, y_iterable):
    yield from x_iterable
    yield from y_iterable

def the_same_chain(x_iterable, y_iterable):
    for x in x_iterable:
        yield x

    for y in y_iterable:
        yield y


a = [1, 2, 3]
b = (4, 5)
for x in chain(a, b):
    print(x)
