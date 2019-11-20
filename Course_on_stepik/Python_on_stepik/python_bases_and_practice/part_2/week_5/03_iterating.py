x = [
    ('Guido', 'Van', 'Rossum'),
    ('Haskell', 'Curry'),
    ('john', 'Backus')
]

# def length(name):
#     return len(' '.join(name))

# name_length = [length(name) for name in x]
# print(name_length)

# x.sort(key=lambda name: len(' '.join(name)))

# x.sort(key=length)
# print(x)

# import operator as op

# x.sort(key=op.itemgetter(-1))
# print(x)

import operator as op
from functools import partial

sort_by_last = partial(list.sort, key=op.itemgetter(-1))
print(x)
sort_by_last(x)
print(x)

y = ['abc', 'cba', 'abb']
sort_by_last(y)
print(y)