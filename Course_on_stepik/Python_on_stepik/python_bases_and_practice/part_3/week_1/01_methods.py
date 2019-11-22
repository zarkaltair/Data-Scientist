print('abc' in 'abcda')
print('abce' in 'abcda')

print('cabcd'.find('abc'))

print(str.find.__doc__)

print('cabc'.index('abc'))
# print('cabc'.index('abcd'))

s = 'The man in black fled across the desert'
print(s.startswith('The man'))
print(s.startswith(('The man', 'The woman', 'The dog')))
print(str.startswith.__doc__)

s = 'image.png'
print(s.endswith('.png'))

s = 'abracadabra'
print(s.count('abra'))
print(str.count.__doc__)

s = 'abacaba'
print(s.find('aba'))
print(s.rfind('aba'))


s = 'The man in black fled across the desert, and the gunslinger followed'
print(s.lower())
print(s.upper())
print(s.count('the'))
print(s.lower().count('the'))

s = '1,2,3,4'
print(s)
print(s.replace(',', ', '))
print(s.replace.__doc__)
print(s.replace(',', ', ', 2))

s = '1 2 3 4'
print(s.split(' '))
print(s.split.__doc__)
print(s.split(' ', 2))

s = '   1 2 3 4         '
print(repr(s.rstrip()))
print(repr(s.lstrip()))
print(repr(s.strip()))

s = '_*__1, 2, 3, 4__*_'
print(repr(s.rstrip('*_')))
print(repr(s.lstrip('*_')))
print(repr(s.strip('*_')))

numbers = map(str, [1, 2, 3, 4, 5])
print(repr(' '.join(numbers)))

capital = 'London is the capital of Great Britain'
template = '{} is the capital of {}'
print(template.format('London', 'Great Britain'))
print(template.format.__doc__)

template = '{capital} is the capital of {country}'
print(template.format(capital='London', country='Great Britain'))
print(template.format(country='Liechtenstein', capital='Vaduz'))

import requests

template = 'Response from {0.url} with code {0.status_code}'

res = requests.get('https://docs.python.org/3.8')
print(template.format(res))

res = requests.get('https://docs.python.org/3.8/random')
print(template.format(res))

from random import random

x = random()
print(x)
print('{:.3}'.format(x))